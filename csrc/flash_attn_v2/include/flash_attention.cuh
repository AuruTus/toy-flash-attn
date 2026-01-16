#pragma once

#include <torch/torch.h>

// FlashForwardKernelConfig contains the configuration for a kernel.
// For choosing the kernel configuration at runtime, We use a map of kernel
// configs to kernels. The official repo uses static switches, which is cleaner
// and faster.
struct FlashForwardKernelConfig {
    const torch::ScalarType dtype;

    const int d_head;   // value = [64, 128]
    const int B_r;      // value = [64, 128]
    const int B_c;      // value = [32, 64, 128]
    const int n_warps;  // value = [4, 8]. 8 only when B_r == 128

    const bool async_copy;
    // If true, load K and V block tiles into smem as soon as we can.
    const bool eager_load_blocks;
    const bool swizzled;

    const int Q_mma_load_K_fragments;
    const int K_mma_load_K_fragments;
    const int V_mma_load_K_fragments;

    // if true, call ldmatrix for the next iter before calling mma.
    const bool mma_double_buffer_loads;
    const bool optimized_softmax;

    int smem_bytes(int elem_size = 2) const {
        return (this->B_r + this->B_c * 2) * this->d_head * elem_size;
    }

    int num_ctas_per_sm(int max_smem_bytes) const {
        if ((this->n_warps == 8) || (max_smem_bytes < smem_bytes() * 2)) {
            return 1;
        }
        return 2;
    }

    bool operator<(const FlashForwardKernelConfig& other) const {
        if (this->dtype != other.dtype) {
            return this->dtype < other.dtype;
        } else if (this->d_head != other.d_head) {
            return this->d_head < other.d_head;
        } else if (this->B_r != other.B_r) {
            return this->B_r < other.B_r;
        } else if (this->B_c != other.B_c) {
            return this->B_c < other.B_c;
        } else if (this->n_warps != other.n_warps) {
            return this->n_warps < other.n_warps;
        } else if (this->async_copy != other.async_copy) {
            return this->async_copy < other.async_copy;
        } else if (this->eager_load_blocks != other.eager_load_blocks) {
            return this->eager_load_blocks < other.eager_load_blocks;
        } else if (this->swizzled != other.swizzled) {
            return this->swizzled < other.swizzled;
        } else if (this->Q_mma_load_K_fragments !=
                   other.Q_mma_load_K_fragments) {
            return this->Q_mma_load_K_fragments < other.Q_mma_load_K_fragments;
        } else if (this->K_mma_load_K_fragments !=
                   other.K_mma_load_K_fragments) {
            return this->K_mma_load_K_fragments < other.K_mma_load_K_fragments;
        } else if (this->V_mma_load_K_fragments !=
                   other.V_mma_load_K_fragments) {
            return this->V_mma_load_K_fragments < other.V_mma_load_K_fragments;
        } else if (this->mma_double_buffer_loads !=
                   other.mma_double_buffer_loads) {
            return this->mma_double_buffer_loads <
                   other.mma_double_buffer_loads;
        } else if (this->optimized_softmax != other.optimized_softmax) {
            return this->optimized_softmax < other.optimized_softmax;
        }

        return false;
    }
};
