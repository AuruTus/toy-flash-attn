import pytest
import torch

from toy_attn.flash_attn_v2 import flash_attention
from toy_attn.flash_attn_v2.kernel_configs import DType, get_kernels_to_build
from utils import (
    BATCH_SIZE_FOR_SEQ_LEN,
    BENCHMARK_N_HEADS,
    QKVConfig,
    generate_qkv,
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
