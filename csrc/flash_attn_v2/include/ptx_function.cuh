#pragma once

#include "common.h"

namespace flash_attn_v2 {

template <typename value_t>
    requires(
        std::is_same_v<value_t, half> || std::is_same_v<value_t, nv_bfloat16>
    )
__device__
void mma_m16n8k16_f32_accum(
    float& d1,
    float& d2,
    float& d3,
    float& d4,

    const uint32_t& a1,
    const uint32_t& a2,
    const uint32_t& a3,
    const uint32_t& a4,

    const uint32_t& b1,
    const uint32_t& b2,

    const float& c1,
    const float& c2,
    const float& c3,
    const float& c4
) {
    if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
                     "{%0, %1, %2, %3}, "
                     "{%4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%10, %11, %12, %13};\n"
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                     "{%0, %1, %2, %3}, "
                     "{%4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%10, %11, %12, %13};\n"
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

__device__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int ngroups>
__device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(ngroups));
}

template <int size, typename T>
__device__
void cp_async(T* smem_to, T* gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                 "l"(gmem_from), "n"(size));
}

template <typename T>
__device__
void ldmatrix_x4(
    T* load_from,
    uint32_t& a1,
    uint32_t& a2,
    uint32_t& a3,
    uint32_t& a4
) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename T>
__device__
void ldmatrix_x4_transpose(
    T* load_from,
    uint32_t& a1,
    uint32_t& a2,
    uint32_t& a3,
    uint32_t& a4
) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

} // namespace flash_attn_v2