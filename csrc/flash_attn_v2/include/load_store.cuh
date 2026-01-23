#pragma once

#include <cuda.h>
#include "common.h"
#include "concepts.h"
#include "ptx_function.cuh"
#include "swizzling.cuh"

namespace flash_attn_v2 {

struct LDSTCommon {
    const bool swizzled;
    const bool async_copy;
};

struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};

// LD/ST configuration for a block (Q, K, V, or O) from GMEM to SMEM
// and vice versa, and also loading from SMEM to RF
struct TensorLDSTConfig {
    // Tile layout for SMEM and RF
    const TileLayout GSM;
    const TileLayout RF;
    const LDSTCommon Common;

    // Block specific properties
    const bool transposed;
    const int block_size;
    const int smem_cols;

    // number of rows a warp in a thread block independently loads/stores,
    // which is equivalent to GSM.row_fragments * 8
    const int warp_ldst_rows;
    // Whether or not the warp will compute over the entire block.
    // This is `false` for (Q & O & S) and `true` for (K & V)
    const bool compute_over_entire_blocks;

    const bool load_entire_block_into_rf;
    const int mma_load_stages;
};

template <typename T>
struct GM2SM_async {
    __device__ static void run(T* gmem, T* smem) {
        cp_async<BYTES_PER_VEC4_ACCESS, T>(smem, gmem);
    }
};

template <typename T>
struct GM2SM {
    __device__ static void run(T* gmem, T* smem) {
        reinterpret_cast<uint4*>(smem)[0] = reinterpret_cast<uint4*>(gmem)[0];
    }
};

template <typename T>
struct SM2GM {
    __device__ static void run(T* gmem, T* smem) {
        reinterpret_cast<uint4*>(gmem)[0] = reinterpret_cast<uint4*>(smem)[0];
    }
};

template <
    typename OP,
    TensorLDSTConfig CFG,
    typename value_t,
    typename index_t = int64_t,
    int WARP_SIZE    = 32
>
    requires gmem_smem_op<OP, value_t>
FA_DEVICE void copy_block_GSM(
    value_t* gmem,
    value_t* smem,
    index_t gmem_seq_stride,
    const int lane_id
) {
    constexpr auto n_row_iters =
        CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    constexpr auto col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr auto col_fragments_per_row  = CFG.smem_cols / COLS_PER_FRAGMENT;

    const auto thread_row          = lane_id / col_fragments_per_iter;
    const auto thread_col_fragment = lane_id % col_fragments_per_iter;

    FA_UNROLL
    for (int r = 0; r < n_row_iters; ++r) {
        const auto curr_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        FA_UNROLL
        for (int c = 0; c < col_fragments_per_row;
             c += col_fragments_per_iter) {
            const auto gmem_col_fragment = c + thread_col_fragment;
            const auto smem_col_fragment = get_smem_col_fragment<
                col_fragments_per_row, CFG.Common.swizzled
            >(curr_row, gmem_col_fragment);

            OP::run(
                &gmem
                    [curr_row * gmem_seq_stride +
                     gmem_col_fragment * COLS_PER_FRAGMENT],
                &smem
                    [curr_row * CFG.smem_cols +
                     smem_col_fragment * COLS_PER_FRAGMENT]
            );
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t, int WARP_SIZE = 32>
FA_DEVICE void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t* smem,
    const int lane_id,
    const int col_fragment_offset = 0
) {
    constexpr auto row_fragments_per_iter = 2;
    constexpr auto rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr auto col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr auto col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const auto thread_row          = lane_id % rows_per_iter;
    const auto thread_col_fragment = lane_id / rows_per_iter;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const auto curr_row = thread_row + r * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const auto smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    curr_row, thread_col_fragment + c + col_fragment_offset
                );

            ldmatrix_x4(
                &smem
                    [curr_row * CFG.smem_cols +
                     smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[r][c], regs[r + 1][c], regs[r][c + 1], regs[r + 1][c + 1]
            );
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t, int WARP_SIZE = 32>
FA_DEVICE void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t* smem,
    const int lane_id,
    const int row_fragment_offset = 0
) {
    constexpr auto row_fragments_per_iter = 2;
    constexpr auto rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr auto col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr auto col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const auto thread_row          = lane_id % rows_per_iter;
    const auto thread_col_fragment = lane_id / rows_per_iter;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const auto curr_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const auto smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    curr_row, thread_col_fragment + c
                );

            ldmatrix_x4_transpose(
                &smem
                    [curr_row * CFG.smem_cols +
                     smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]
            );
        }
    }
}

template <TensorLDSTConfig CFG, typename value_t, int WARP_SIZE = 32>
FA_DEVICE void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments],
    value_t* smem,
    const int lane_id
) {
    constexpr auto rows_per_iter          = ROWS_PER_FRAGMENT;
    constexpr auto col_fragments_per_iter = 1;
    constexpr auto col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;

    constexpr auto elems_per_store = 2;
    const auto thread_row          = lane_id / 4;
    const auto thread_inner_col    = (lane_id % 4) * elems_per_store;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const auto curr_row = r * rows_per_iter + thread_row;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const auto smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    curr_row, c
                );
            reinterpret_cast<uint32_t*>(
                &smem
                    [curr_row * CFG.smem_cols +
                     (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                      thread_inner_col)]
            )[0] = regs[r][c];
        }
    }
}

template <
    typename value_t,
    int M_fragments,
    int N_fragments,
    bool is_half = std::is_same_v<value_t, half>
>
FA_DEVICE_CONSTEXPR void convert_to_16_bit_dtype(
    float (&src_float)[M_fragments][N_fragments * 2],
    uint32_t (&dest_uint)[M_fragments][N_fragments]
) {
    using value2_t = std::conditional_t<is_half, half2, nv_bfloat162>;

    auto src = reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    auto dest =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    FA_UNROLL
    for (int m = 0; m < M_fragments; ++m) {
        FA_UNROLL
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (is_half) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}

}  // namespace flash_attn_v2