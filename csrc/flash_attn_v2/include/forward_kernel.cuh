#pragma once

#include "common.h"
#include "concepts.h"

#include "gemm.cuh"
#include "load_store.cuh"
#include "ptx_function.cuh"
#include "softmax.cuh"
#include "static_kernel_config.cuh"

namespace flash_attn_v2 {

struct ForwardKernelArgs {
    using index_t = int64_t;

    void* __restrict__ Q;
    void* __restrict__ K;
    void* __restrict__ V;
    void* __restrict__ O;

    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_Q_blocks;
    const int n_KV_blocks;
};

// clang-format off
/**
 * Flash Attention Forward Kernel
 *
 * Tile Shapes (per thread block):
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Tensor │ GMEM Shape         │ SMEM Tile      │ Description              │
 * ├────────┼────────────────────┼────────────────┼──────────────────────────┤
 * │ Q      │ (seq, d_head)      │ (B_r, d_head)  │ Query block              │
 * │ K      │ (seq, d_head)      │ (B_c, d_head)  │ Key block (streamed)     │
 * │ V      │ (seq, d_head)      │ (B_c, d_head)  │ Value block (streamed)   │
 * │ O      │ (seq, d_head)      │ (B_r, d_head)  │ Output block (reuses Q)  │
 * │ S      │ -                  │ RF only        │ QK^T scores (B_r, B_c)   │
 * │ P      │ -                  │ RF only        │ softmax(S)  (B_r, B_c)   │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * SMEM Layout:
 * ┌──────────────────┬──────────────────┬──────────────────┐
 * │  Q / O           │       K          │       V          │
 * │  (B_r × d_head)  │  (B_c × d_head)  │  (B_c × d_head)  │
 * └──────────────────┴──────────────────┴──────────────────┘
 * Note: Q and O share the same SMEM region (non-overlapping lifetimes)
 *
 * Per-Warp Register Tiles (see static_kernel_config.cuh for RF layout):
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Tensor  │ Logical Shape               │ Notes                           │
 * ├─────────┼─────────────────────────────┼─────────────────────────────────┤
 * │ Q_rf    │ (B_r/n_warps, d_head)       │ Each warp owns B_r/n_warps rows │
 * │ K_rf    │ (B_c, d_head)               │ Full K block (all warps share)  │
 * │ V_rf    │ (d_head, B_c)               │ Transposed for P @ V matmul     │
 * │ S_accum │ (B_r/n_warps, B_c)          │ float32 accumulator             │
 * │ P_b16   │ (B_r/n_warps, B_c)          │ fp16/bf16 after softmax         │
 * │ O_accum │ (B_r/n_warps, d_head)       │ float32 accumulator             │
 * │ O_b16   │ (B_r/n_warps, d_head)       │ fp16/bf16 for output            │
 * │ m, l    │ (B_r/n_warps,)              │ Softmax state (row max & sum)   │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Computation Flow:
 *   for each KV block j:
 *       S[B_r, B_c] = Q[B_r, d] @ K[B_c, d]^T    // Attention scores
 *       P[B_r, B_c] = online_softmax(S)          // Row-wise softmax
 *       O[B_r, d]  += P[B_r, B_c] @ V[B_c, d]    // Weighted sum
 *   O = O / l                                    // Final normalization
 */
// clang-format on
template <typename Kernel>
    requires kernel_trait<Kernel>
__global__ void flash_forward_kernel(
    __grid_constant__ const ForwardKernelArgs args
) {
    using accum_t = float;

    using index_t = typename Kernel::index_t;
    using value_t = typename Kernel::value_t;

    using N         = typename Kernel::N;
    using Q_t       = typename Kernel::Q_t;
    using K_t       = typename Kernel::K_t;
    using V_t       = typename Kernel::V_t;
    using S_accum_t = typename Kernel::S_accum_t;
    using P_value_t = typename Kernel::P_value_t;
    using O_accum_t = typename Kernel::O_accum_t;
    using O_value_t = typename Kernel::O_value_t;

    constexpr bool async_copy = Kernel::async_copy;

    const int sample      = blockIdx.z;
    const int head        = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;
    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;
    const index_t KV_gmem_block_offset = sample_head_offset;

    value_t* gmem_Q = &static_cast<value_t*>(args.Q)[QO_gmem_block_offset];
    value_t* gmem_O = &static_cast<value_t*>(args.O)[QO_gmem_block_offset];
    value_t* gmem_K = &static_cast<value_t*>(args.K)[KV_gmem_block_offset];
    value_t* gmem_V = &static_cast<value_t*>(args.V)[KV_gmem_block_offset];

    extern __shared__ __align__(16) char ch_smem[];
    value_t* smem_Q = reinterpret_cast<value_t*>(ch_smem);
    value_t* smem_O = smem_Q;
    value_t* smem_K = &smem_Q[Kernel::B_r * Kernel::d_head];
    value_t* smem_V = &smem_K[Kernel::B_c * Kernel::d_head];

    // Pointers to the K&V locations in smem that the warp copies to.
    auto Q = Q_t(gmem_Q, gmem_seq_stride, smem_Q);
    auto K = K_t(gmem_K, gmem_seq_stride, smem_K);
    auto V = V_t(gmem_V, gmem_seq_stride, smem_V);

    // S and P are only stored in registers (RF only, no SMEM).
    // S_accum: (B_r/n_warps, B_c) per warp, float32
    // P_b16:   (B_r/n_warps, B_c) per warp, fp16/bf16
    auto S_accum = S_accum_t(nullptr, -1, nullptr);
    auto P_b16   = P_value_t(nullptr, -1, nullptr);
    // The accumulator for O is only kept in registers.
    // O_accum: (B_r/n_warps, d_head) per warp, float32
    // O_b16:   (B_r/n_warps, d_head) per warp, fp16/bf16
    // At the end of the kernel, it is then converted into a 16-bit type and
    // then copied into gmem.
    auto O_accum = O_accum_t(nullptr, -1, nullptr);
    auto O_b16   = O_value_t(gmem_O, gmem_seq_stride, smem_O);

    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit<async_copy>();
    if constexpr (Kernel::eager_load_blocks) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<async_copy>();
    }

    O_accum.zero();

    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(Kernel::d_head)) *
                                  (Kernel::optimized_softmax ? M_LOG2E : 1.0f);
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<accum_t>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    FA_UNROLL
    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0f;
    }

    if constexpr (Q_t::load_entire_block_into_rf) {
        if constexpr (Kernel::eager_load_blocks) {
            // We only wait for the Q block to finish loading.
            cp_async_wait<1, async_copy>();
        } else {
            cp_async_wait<0, async_copy>();
        }

        // We need the __syncwarp() in addition to the cp_async_wait()
        // because cp_async_wait() only blocks until the current thread has
        // finished loading. The entire warp will read this block from
        // smem, so we need to wait on a warp-wide barrier.
        // For K and V, we will need a __syncthread() instead.
        __syncwarp();
        Q.copy_SM2RF();  // Q[B_r, d_head] SMEM -> RF (each warp: B_r/n_warps
                         // rows)
    }

    for (int j = 0; j < args.n_KV_blocks; ++j) {
        if constexpr (!Kernel::eager_load_blocks) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit<async_copy>();
        }
        // Initialize the registers for S to 0.
        S_accum.zero();

        // Block until we've copied the K block-tile for this iteration into
        // shared memory.
        cp_async_wait<0, async_copy>();
        // After this barrier, it is safe to load the next V block, because all
        // warps have done the previous PV matmul.
        __syncthreads();

        if constexpr (Kernel::eager_load_blocks) {
            // Start the (async) copy for the V matrix from gmem to smem but
            // do not wait until after the S=QK matmul.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async_copy>();
        }
        if constexpr (K_t::load_entire_block_into_rf) {
            K.copy_SM2RF();  // K[B_c, d_head] SMEM -> RF (full block, all
                             // warps)
        }

        // S[B_r, B_c] = Q[B_r, d_head] @ K[B_c, d_head]^T
        matmul<Kernel::S_QK_GEMM>(Q, K, S_accum);
        cp_async_wait<0, async_copy>();
        // After this barrier, it is safe to load the next block of K.
        __syncthreads();

        if constexpr (Kernel::eager_load_blocks) {
            // Start the async copy for the next K block-tile from gmem to
            // smem, but do not wait for the copy until the next iteration
            // when we need it.
            if (j < args.n_KV_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<async_copy>();
            }
        }

        // Online softmax: P[B_r, B_c] = softmax(S[B_r, B_c])
        // m: row max, l: row sum (both shape: B_r/n_warps per warp)
        accum_t m_next[N::QO_fragments_per_warp];
        if constexpr (!Kernel::optimized_softmax) {
            scale_S_accum(S_accum.data(), softmax_scale);  // S *= scale
        }
        calc_row_max(S_accum.data(), m_next, m);  // m_next = max(m, rowmax(S))
        scale_l_O<
            Kernel::optimized_softmax
        >(                                               // l *= exp(m - m_next)
            m_next, m, l, O_accum.data(), softmax_scale  // O *= exp(m - m_next)
        );

        exponentiate_tensor<
            Kernel::optimized_softmax
        >(  // S = exp(S - m_next)
            S_accum.data(), m_next, softmax_scale
        );

        update_row_exp_sum(S_accum.data(), l);  // l += rowsum(S)

        // Convert S (float32) -> P (fp16/bf16) for MMA input
        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());

        if constexpr (!Kernel::eager_load_blocks) {
            // Load V from gmem to smem and block until it is done.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async_copy>();
            cp_async_wait<0, async_copy>();
            __syncthreads();
        }

        if constexpr (V_t::load_entire_block_into_rf) {
            V.copy_SM2RF();  // V[B_c, d_head] -> RF
        }

        // O[B_r, d_head] += P[B_r, B_c] @ V[B_c, d_head]
        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
    }

    // O = O / l (final row-wise normalization)
    final_softmax_normalization(O_accum.data(), l);
    // Convert O (float32) -> O (fp16/bf16) for output
    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());

    // Instead of writing directly to gmem, we write to smem as an intermediary
    // step. This allows us to
    // - use 16B vectorized stores, as opposed to 4B stores
    // - fully coalesce our stores
    //   - each warp can store 4x128B aligned lines (512B/warp) instead
    //   of 8x16B uncoalesced rows (128B/warp)
    O_b16.copy_RF2SM();

    // Wait until all threads in the same warp have written to smem.
    // We do not need __syncthreads() here because the warps operate on
    // independent chunks of O.
    __syncwarp();

    // Copy the final O tile from smem to gmem.
    O_b16.copy_SM2GM();
}
}  // namespace flash_attn_v2