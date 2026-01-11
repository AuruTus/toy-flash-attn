#pragma once

#include <concepts>
#include <type_traits>

#include "common.h"

namespace flash_attn_v2 {

template <typename T, typename value_t>
concept gmem_smem_op = requires(value_t* gmem, value_t* smem) {
    { T::run(gmem, smem) } -> std::same_as<void>;
};

template <typename T, typename value_t, typename index_t = int64_t>
concept ldst_trait = requires {
    requires requires(
        value_t* gmem_ptr, index_t gmem_seq_stride, value_t* smem_ptr
    ) {
        { T(gmem_ptr, gmem_seq_stride, smem_ptr) };
    };

    requires requires(T t) {
        { t.copy_GM2SM() } -> std::same_as<void>;
        { t.copy_SM2RF() } -> std::same_as<void>;
        { t.copy_SM2GM() } -> std::same_as<void>;
        { t.advance_gmem_block() } -> std::same_as<void>;
    };

    { T::load_entire_block_into_rf } -> std::convertible_to<bool>;
    { T::mma_load_stages } -> std::convertible_to<int>;
};

template <typename T>
concept fragment_trait = requires {
    { T::QO_fragments_per_warp } -> std::convertible_to<int64_t>;
};

template <typename Kernel>
concept kernel_trait = requires {
    typename Kernel::N;

    typename Kernel::value_t;
    typename Kernel::index_t;

    requires std::convertible_to<typename Kernel::index_t, int64_t>;

    typename Kernel::Q_t;
    typename Kernel::K_t;
    typename Kernel::V_t;
    typename Kernel::S_accum_t;
    typename Kernel::P_value_t;
    typename Kernel::O_accum_t;
    typename Kernel::O_value_t;

    // clang-format off
    requires ldst_trait<typename Kernel::Q_t, typename Kernel::value_t, typename Kernel::index_t>;
    requires ldst_trait<typename Kernel::K_t, typename Kernel::value_t, typename Kernel::index_t>;
    requires ldst_trait<typename Kernel::V_t, typename Kernel::value_t, typename Kernel::index_t>;
    requires ldst_trait<typename Kernel::S_accum_t, typename Kernel::value_t, typename Kernel::index_t>;
    requires ldst_trait<typename Kernel::P_value_t, typename Kernel::value_t, typename Kernel::index_t>;
    requires ldst_trait<typename Kernel::O_accum_t, typename Kernel::value_t, typename Kernel::index_t>;
    requires ldst_trait<typename Kernel::O_value_t, typename Kernel::value_t, typename Kernel::index_t>;
    // clang-format on

    typename Kernel::S_QK_GEMM;

    { Kernel::async } -> std::convertible_to<bool>;
    { Kernel::B_r } -> std::convertible_to<typename Kernel::index_t>;
    { Kernel::B_c } -> std::convertible_to<typename Kernel::index_t>;
    { Kernel::d_head } -> std::convertible_to<typename Kernel::index_t>;
    { Kernel::optimized_softmax } -> std::convertible_to<bool>;
    { Kernel::eager_load_blocks } -> std::convertible_to<bool>;
};

}  // namespace flash_attn_v2