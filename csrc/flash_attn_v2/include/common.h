#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace flash_attn_v2 {

#define FA_UNROLL _Pragma("unroll")

inline constexpr int WARP_SIZE_DEFAULT = 32;

inline constexpr int SHFL_ENTIRE_WARP_MASK = 0xffffffff;

inline constexpr int N_REGS_PER_F32_ACCUM_FRAGMENT = 2;

// MMA register layout parameters
inline constexpr int MMA_A_REGS_PER_ROW = 2;
inline constexpr int MMA_A_REGS_PER_COL = 2;
inline constexpr int MMA_B_REGS_PER_ROW = 2;
inline constexpr int MMA_B_REGS_PER_COL = 1;
inline constexpr int MMA_C_REGS_PER_ROW = 1;
inline constexpr int MMA_C_REGS_PER_COL = 2;

// Load/store parameters
inline constexpr int B16_BYTES             = 2;
inline constexpr int BYTES_PER_VEC4_ACCESS = 16;
inline constexpr int ELEMS_PER_VEC4_ACCESS =
    (BYTES_PER_VEC4_ACCESS / B16_BYTES);

inline constexpr int LDMATRIX_MAT_SIZE = 8;
inline constexpr int ROWS_PER_FRAGMENT = LDMATRIX_MAT_SIZE;
inline constexpr int COLS_PER_FRAGMENT = LDMATRIX_MAT_SIZE;

inline constexpr int GSM_LDST_ROWS_PER_ITER = 4;

};  // namespace flash_attn_v2
