#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace flash_attn_v2 {

constexpr int WARP_SIZE_DEFAULT = 32;

constexpr int SHFL_ENTIRE_WARP_MASK = 0xffffffff;

constexpr int N_REGS_PER_F32_ACCUM_FRAGMENT = 2;

// MMA register layout parameters
constexpr int MMA_A_REGS_PER_ROW = 2;
constexpr int MMA_A_REGS_PER_COL = 2;
constexpr int MMA_B_REGS_PER_ROW = 2;
constexpr int MMA_B_REGS_PER_COL = 1;
constexpr int MMA_C_REGS_PER_ROW = 1;
constexpr int MMA_C_REGS_PER_COL = 2;

// Load/store parameters
constexpr int B16_BYTES = 2;
constexpr int BYTES_PER_VEC4_ACCESS = 16;
constexpr int ELEMS_PER_VEC4_ACCESS = (BYTES_PER_VEC4_ACCESS / B16_BYTES);

constexpr int LDMATRIX_MAT_SIZE = 8;
constexpr int ROWS_PER_FRAGMENT = LDMATRIX_MAT_SIZE;
constexpr int COLS_PER_FRAGMENT = LDMATRIX_MAT_SIZE;

constexpr int GSM_LDST_ROWS_PER_ITER = 4;

}; // namespace flash_attn_v2