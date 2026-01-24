#pragma once

#include <type_traits>

#include "load_store.cuh"

namespace flash_attn_v2 {
template <typename value_t, int N>
struct RFVector {
    static constexpr auto size = N;
    value_t regs[N];

    FA_DEVICE_CONSTEXPR value_t& operator[](int idx) { return regs[idx]; }
};

template <typename value_t, int n_copies, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;

    static constexpr auto regs_per_fragment = sizeof(value_t) / 2;
    static constexpr auto rows              = row_fragments;
    static constexpr auto cols              = col_fragments * regs_per_fragment;

    storage_t regs[n_copies][rows][cols];

    FA_DEVICE_CONSTEXPR auto data(const int stage = 0)
        -> storage_t (&)[rows][cols] {
        return reinterpret_cast<storage_t(&)[rows][cols]>(regs[stage]);
    }

    FA_DEVICE_CONSTEXPR void zero() {
        FA_UNROLL
        for (int i = 0; i < n_copies; ++i) {
            FA_UNROLL
            for (int j = 0; j < rows; ++j) {
                FA_UNROLL
                for (int k = 0; k < cols; ++k) {
                    this->regs[i][j][k] = 0;
                }
            }
        }
    }
};

// MatrixLDST is an object that provides ldst and conversion functionality for a
// block in memory. The scope of the object involves all the levels of memory
// (gmem, smem, and the rf). Admittedly, this class does too much, but I didn't
// want to overengineer it given the scope of this project.
template <
    TensorLDSTConfig ldst,
    typename value_t,
    typename index_t = int64_t,
    int WARP_SIZE    = WARP_SIZE_DEFAULT
>
struct MatrixLDST {
    using matrix_storage_t = RFMatrix<
        value_t,
        ldst.mma_load_stages,
        ldst.RF.row_fragments,
        ldst.RF.col_fragments
    >;

    using GM2SM_op = std::conditional_t<
        ldst.Common.async_copy,
        GM2SM_async<value_t>,
        GM2SM<value_t>
    >;

    using SM2GM_op = SM2GM<value_t>;

    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf =
        ldst.load_entire_block_into_rf;
    static constexpr bool transposed = ldst.transposed;

    // Runtime properties
    value_t* gmem_ptr;
    index_t gmem_seq_stride;
    // The location in memory used to load fragments from SMEM to RF
    value_t* smem_srm_ptr;
    // The location in memory that the warp writes to for Q, K, V from GMEM to
    // SMEM and O for SMEM to GMEM
    value_t* smem_gsm_ptr;

    const int lane_id;

    matrix_storage_t storage;

    FA_DEVICE MatrixLDST(
        value_t* gmem_block_ptr,
        index_t gmem_seq_stride,
        value_t* smem_ptr
    )
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank    = threadIdx.x / WARP_SIZE;
        const index_t warp_seq = ldst.warp_ldst_rows * warp_rank;

        this->gmem_seq_stride = gmem_seq_stride;
        this->gmem_ptr        = gmem_block_ptr
                                    ? gmem_block_ptr + warp_seq * gmem_seq_stride
                                    : nullptr;
        this->smem_gsm_ptr =
            smem_ptr ? smem_ptr + warp_seq * ldst.smem_cols : nullptr;
        this->smem_srm_ptr =
            ldst.compute_over_entire_blocks ? smem_ptr : smem_gsm_ptr;
    }

    FA_DEVICE_CONSTEXPR void zero() { this->storage.zero(); }

    FA_DEVICE_CONSTEXPR auto data(const int stage = 0) ->
        typename matrix_storage_t::storage_t (&)[matrix_storage_t::rows]
                                                [matrix_storage_t::cols] {
        return this->storage.data(stage);
    }

    FA_DEVICE_CONSTEXPR void advance_gmem_block() {
        this->gmem_ptr += ldst.block_size * this->gmem_seq_stride;
    }

    FA_DEVICE_CONSTEXPR void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst, value_t, index_t, WARP_SIZE>(
            this->gmem_ptr, this->smem_gsm_ptr, this->gmem_seq_stride,
            this->lane_id
        );
    }

    FA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst, value_t, index_t, WARP_SIZE>(
            this->gmem_ptr, this->smem_gsm_ptr, this->gmem_seq_stride,
            this->lane_id
        );
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF(int stage = 0, int tile_offset = 0) {
        if constexpr (!this->transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t, WARP_SIZE>(
                this->storage.data(stage), this->smem_srm_ptr, this->lane_id,
                tile_offset
            );
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t, WARP_SIZE>(
                this->storage.data(stage), this->smem_srm_ptr, this->lane_id,
                tile_offset
            );
        }
    }

    FA_DEVICE_CONSTEXPR void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t, WARP_SIZE>(
            this->data(), this->smem_srm_ptr, this->lane_id
        );
    }
};

}  // namespace flash_attn_v2