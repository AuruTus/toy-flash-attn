import os
import torch
import pytest

from toy_attn.flash_attn_v2 import flash_attention
from toy_attn.flash_attn_v2.kernel_configs import DType, get_kernels_to_build
from utils import (
    BATCH_SIZE_FOR_SEQ_LEN,
    BENCHMARK_N_HEADS,
    QKVConfig,
    generate_qkv,
    make_debug_tensor,
    py_flash_attention,
)


def test_kernel_build():
    """Test that flash attention kernels are built correctly."""
    from toy_attn.flash_attn_v2.kernel import flash_attention_kernels

    assert flash_attention_kernels is not None


# Generate configs at module level
FP16_CONFIGS = [cfg for cfg in get_kernels_to_build() if cfg.dtype == DType.FP16]
BF16_CONFIGS = [cfg for cfg in get_kernels_to_build() if cfg.dtype == DType.BF16]


def _generate_test_data(dtype: torch.dtype, d_heads=[128]):
    """Helper to generate test data for flash attention tests."""
    seq_len = 2048
    batch_size = BATCH_SIZE_FOR_SEQ_LEN[seq_len]
    n_heads = BENCHMARK_N_HEADS
    device = "cuda:0"

    data = {}
    pt_b16_results = {}
    pt_f32_results = {}

    for d_head in d_heads:
        cfg = QKVConfig(
            n_heads=n_heads,
            d_head=d_head,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
        )

        q, k, v = generate_qkv(cfg)
        data[d_head] = (q, k, v)
        pt_b16_results[d_head] = py_flash_attention(q, k, v, upcast=False)
        pt_f32_results[d_head] = py_flash_attention(q, k, v, upcast=True)

    return {
        "data": data,
        "pt_b16_results": pt_b16_results,
        "pt_f32_results": pt_f32_results,
    }


@pytest.mark.parametrize("cfg", FP16_CONFIGS, ids=str)
class TestFlashAttentionFP16:
    @pytest.fixture(scope="class")
    def test_data(self):
        """Generate test data once for all FP16 tests."""
        return _generate_test_data(dtype=torch.float16, d_heads=[128])

    def test_fp16_kernel(self, test_data, cfg):
        """Test FP16 flash attention kernel against PyTorch reference."""
        q, k, v = test_data["data"][cfg.d_head]
        result = flash_attention.forward(cfg, q, k, v)
        fp16_result = test_data["pt_b16_results"][cfg.d_head]
        fp32_result = test_data["pt_f32_results"][cfg.d_head]

        # Based on https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
        diff_fp16 = (result - fp16_result).abs().max().item()
        diff_fp32 = (fp16_result - fp32_result).abs().max().item()

        assert diff_fp16 <= diff_fp32 * 2, (
            f"FP16 kernel difference ({diff_fp16:.6e}) exceeds threshold (2 * FP32 baseline = {diff_fp32 * 2:.6e})"
        )


@pytest.mark.parametrize("cfg", BF16_CONFIGS, ids=str)
class TestFlashAttentionBF16:
    @pytest.fixture(scope="class")
    def test_data(self):
        """Generate test data once for all BF16 tests."""
        return _generate_test_data(dtype=torch.bfloat16, d_heads=[128])

    def test_bf16_kernel(self, test_data, cfg):
        """Test BF16 flash attention kernel against PyTorch reference."""
        q, k, v = test_data["data"][cfg.d_head]
        result = flash_attention.forward(cfg, q, k, v)
        fp16_result = test_data["pt_b16_results"][cfg.d_head]
        fp32_result = test_data["pt_f32_results"][cfg.d_head]

        # Based on https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
        diff_fp16 = (result - fp16_result).abs().max().item()
        diff_fp32 = (fp16_result - fp32_result).abs().max().item()

        assert diff_fp16 <= diff_fp32 * 2, (
            f"BF16 kernel difference ({diff_fp16:.6e}) exceeds threshold (2 * FP32 baseline = {diff_fp32 * 2:.6e})"
        )


@pytest.mark.skipif(
    condition=os.environ.get("FA_DEBUG", "false").lower() != "true",
    reason="Debug entry. Skip for normal tests",
)
def test_debug_nan_bug_kernel():
    """Debug test for previous NaN bug with minimal tensor sizes."""
    from toy_attn.flash_attn_v2.kernel_configs import FlashForwardKernelConfig

    # Use smallest valid configuration
    # seq_len must be divisible by B_r and B_c
    B_r, B_c = 64, 64
    d_head = 128
    batch_size = 1
    n_heads = 1
    seq_len = 128  # Must be >= max(B_r, B_c) and divisible by both

    device = "cuda:0"
    dtype = torch.float16

    # Create minimal config matching a built kernel
    # B_r=64, B_c=64 with load_0_0_0 pattern (simplest)
    cfg = FlashForwardKernelConfig(
        dtype=DType.FP16,
        d_head=d_head,
        B_r=B_r,
        B_c=B_c,
        n_warps=4,
        async_copy=True,  # Required by built kernels
        eager_load_blocks=True,  # Required by built kernels
        swizzled=False,  # Required by built kernels
        Q_mma_load_K_tiles=0,  # Load entire block (simplest)
        K_mma_load_K_tiles=0,
        V_mma_load_K_tiles=0,
        mma_double_buffer_loads=False,
        optimized_softmax=False,
    )

    print(f"\n=== Debug Test ===")
    print(f"Config: {cfg.short_form()}")
    print(f"Tensor shape: ({batch_size}, {seq_len}, {n_heads}, {d_head})")
    print(f"B_r={B_r}, B_c={B_c}, n_Q_blocks={seq_len // B_r}, n_KV_blocks={seq_len // B_c}")

    # Calculate expected smem usage
    smem_bytes = (B_r + B_c * 2) * d_head * 2  # 2 bytes per element
    print(f"Expected smem: {smem_bytes} bytes ({smem_bytes / 1024:.1f} KB)")

    q = make_debug_tensor(batch_size, seq_len, n_heads, d_head, dtype, device)
    k = make_debug_tensor(batch_size, seq_len, n_heads, d_head, dtype, device)
    v = make_debug_tensor(batch_size, seq_len, n_heads, d_head, dtype, device)

    print(f"Q stride: {q.stride()}")
    print(f"Q is contiguous: {q.is_contiguous()}")

    torch.cuda.synchronize()
    print("Tensors created, calling kernel...")

    # Call kernel
    try:
        result = flash_attention.forward(cfg, q, k, v)
        torch.cuda.synchronize()
        print(f"Kernel completed! Result shape: {result.shape}")

        # Check for NaN/Inf in output
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"Result has NaN: {has_nan}, has Inf: {has_inf}")

        if has_nan or has_inf:
            # Find where NaN/Inf occurs
            nan_mask = torch.isnan(result) | torch.isinf(result)
            nan_indices = nan_mask.nonzero()
            print(f"First few NaN/Inf indices: {nan_indices[:5].tolist()}")
            print(f"Result sample (first few): {result.flatten()[:10].tolist()}")

        # Compare with reference
        expected = py_flash_attention(q, k, v, upcast=True)
        diff = (result - expected).abs().max().item()
        print(f"Max diff from reference: {diff:.6e}")

        assert result.shape == q.shape, "Output shape mismatch"
        assert not has_nan, "Output contains NaN"
        assert not has_inf, "Output contains Inf"

    except Exception as e:
        print(f"Kernel failed with: {type(e).__name__}: {e}")
        raise


@pytest.mark.skipif(
    condition=os.environ.get("FA_DEBUG", "false").lower() != "true",
    reason="Debug entry. Skip for normal tests",
)
def test_debug_double_buffer():
    """Debug test for double buffer implementation (mma_double_buffer_loads=True).

    Uses load_2_2_2_tiles config which actually triggers double buffering.
    When Q/K/V_mma_load_K_tiles > 0 AND mma_double_buffer_loads=True,
    the mma_load_stages becomes 2, enabling double buffering.

    Uses B_r=64, B_c=32 because for B_r=64, B_c=64, configs with
    Q_mma_load_K_tiles != 0 are excluded by should_autotune_config().
    """
    from toy_attn.flash_attn_v2.kernel_configs import FlashForwardKernelConfig

    B_r, B_c = 64, 32  # B_c=32 to allow Q_mma_load_K_tiles=2
    d_head = 128
    batch_size = 1
    n_heads = 1
    seq_len = 128

    device = "cuda:0"
    dtype = torch.float16

    # Config with double buffer ACTUALLY enabled (load_2_2_2_tiles+buffer)
    # Q/K/V_mma_load_K_tiles=2 means load 2 K-tiles per iteration during MMA
    # mma_double_buffer_loads=True means use 2 stages for these loads
    cfg = FlashForwardKernelConfig(
        dtype=DType.FP16,
        d_head=d_head,
        B_r=B_r,
        B_c=B_c,
        n_warps=4,
        async_copy=True,
        eager_load_blocks=True,
        swizzled=False,
        Q_mma_load_K_tiles=2,  # Non-zero to trigger double buffer
        K_mma_load_K_tiles=2,  # Non-zero to trigger double buffer
        V_mma_load_K_tiles=2,  # Non-zero to trigger double buffer
        mma_double_buffer_loads=True,  # Double buffer enabled
        optimized_softmax=False,
    )

    print(f"\n=== Debug Double Buffer Test ===")
    print(f"Config: {cfg.short_form()}")
    print(f"Tensor shape: ({batch_size}, {seq_len}, {n_heads}, {d_head})")
    print(f"B_r={B_r}, B_c={B_c}, n_Q_blocks={seq_len // B_r}, n_KV_blocks={seq_len // B_c}")

    q = make_debug_tensor(batch_size, seq_len, n_heads, d_head, dtype, device)
    k = make_debug_tensor(batch_size, seq_len, n_heads, d_head, dtype, device)
    v = make_debug_tensor(batch_size, seq_len, n_heads, d_head, dtype, device)

    print(f"Q stride: {q.stride()}")
    print(f"Q is contiguous: {q.is_contiguous()}")

    torch.cuda.synchronize()
    print("Tensors created, calling kernel...")

    try:
        result = flash_attention.forward(cfg, q, k, v)
        torch.cuda.synchronize()
        print(f"Kernel completed! Result shape: {result.shape}")

        # Check for NaN/Inf in output
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"Result has NaN: {has_nan}, has Inf: {has_inf}")

        if has_nan or has_inf:
            nan_mask = torch.isnan(result) | torch.isinf(result)
            nan_indices = nan_mask.nonzero()
            print(f"First few NaN/Inf indices: {nan_indices[:5].tolist()}")
            print(f"Result sample (first few): {result.flatten()[:10].tolist()}")

        # Compare with reference
        expected = py_flash_attention(q, k, v, upcast=True)
        diff = (result - expected).abs().max().item()
        print(f"Max diff from reference: {diff:.6e}")

        # Compare with non-buffered baseline (same tile config but no double buffer)
        cfg_no_buffer = FlashForwardKernelConfig(
            dtype=DType.FP16,
            d_head=d_head,
            B_r=B_r,
            B_c=B_c,
            n_warps=4,
            async_copy=True,
            eager_load_blocks=True,
            swizzled=False,
            Q_mma_load_K_tiles=2,
            K_mma_load_K_tiles=2,
            V_mma_load_K_tiles=2,
            mma_double_buffer_loads=False,  # No double buffer
            optimized_softmax=False,
        )
        result_no_buffer = flash_attention.forward(cfg_no_buffer, q, k, v)
        diff_no_buffer = (result_no_buffer - expected).abs().max().item()
        diff_between = (result - result_no_buffer).abs().max().item()
        print(f"Non-buffered max diff from reference: {diff_no_buffer:.6e}")
        print(f"Diff between buffered and non-buffered: {diff_between:.6e}")

        assert result.shape == q.shape, "Output shape mismatch"
        assert not has_nan, "Output contains NaN"
        assert not has_inf, "Output contains Inf"
        assert diff <= diff_no_buffer * 2, (
            f"Double buffer diff ({diff:.6e}) exceeds threshold (2 * non-buffered = {diff_no_buffer * 2:.6e})"
        )

    except Exception as e:
        print(f"Kernel failed with: {type(e).__name__}: {e}")
        raise
