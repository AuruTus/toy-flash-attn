import pytest


def test_kernel_build():
    from toy_attn.flash_attn_v2.kernel import flash_attention_kernels

    assert flash_attention_kernels is not None
