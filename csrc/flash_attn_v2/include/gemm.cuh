#pragma once

#include "common.h"
#include "concepts.h"
#include "ptx_function.cuh"

namespace flash_attn_v2 {

inline constexpr int MMA_M = 16;
inline constexpr int MMA_N = 8;
inline constexpr int MMA_K = 16;

inline constexpr int MMA_M_FRAGMENTS_PER_ITER = MMA_M / LDMATRIX_MAT_SIZE;
inline constexpr int MMA_N_FRAGMENTS_PER_ITER = MMA_N / LDMATRIX_MAT_SIZE;
inline constexpr int MMA_K_FRAGMENTS_PER_ITER = MMA_K / LDMATRIX_MAT_SIZE;

template <
    typename _A_t,
    typename _B_t,
    typename _C_t,
    int total_K_fragments,
    int load_K_fragments_per_iter,
    typename _value_t
>
struct GEMM {
    using A_t     = _A_t;
    using B_t     = _B_t;
    using C_t     = _C_t;
    using value_t = _value_t;

    static constexpr int TotalKTiles       = total_K_fragments;
    static constexpr int LoadKTilesPerIter = load_K_fragments_per_iter;

    static constexpr bool DoubleBufferA =
        !A_t::load_entire_block_into_rf && A_t::mma_load_stages > 1;
    static constexpr bool DoubleBufferB =
        !B_t::load_entire_block_into_rf && B_t::mma_load_stages > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

template <
    typename value_t,
    int M_fragments,
    int N_fragments,
    int K_fragments,
    typename accum_t = float
>
__forceinline__ __device__ void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments],
    uint32_t (&regs_B)[N_fragments][K_fragments],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT]
) {
    FA_UNROLL
    for (int k = 0; k < K_fragments; k += MMA_K_FRAGMENTS_PER_ITER) {
        FA_UNROLL
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            FA_UNROLL
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    // d
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2 + 1], regs_C[m + 1][n * 2 + 1],
                    // a
                    regs_A[m][k], regs_A[m + 1][k], regs_A[m][k + 1],
                    regs_A[m + 1][k + 1],
                    // b
                    regs_B[n][k], regs_B[n][k + 1],
                    // c
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1]
                );
            }
        }
    }
}

template <typename GEMM>
    requires gemm_trait<GEMM>
__forceinline__ __device__ void
matmul(typename GEMM::A_t& A, typename GEMM::B_t& B, typename GEMM::C_t& C) {}

}  // namespace flash_attn_v2
