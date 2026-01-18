"""Flash Attention V2 wrapper module."""

import torch

from toy_attn.flash_attn_v2.kernel import flash_attention_kernels
from toy_attn.flash_attn_v2.kernel_configs import FlashForwardKernelConfig


def forward(
    cfg: FlashForwardKernelConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass for flash attention.

    Args:
        cfg: Kernel configuration
        q: Query tensor of shape (batch, seq_len, n_heads, d_head)
        k: Key tensor of shape (batch, seq_len, n_heads, d_head)
        v: Value tensor of shape (batch, seq_len, n_heads, d_head)

    Returns:
        Output tensor of shape (batch, seq_len, n_heads, d_head)
    """
    return flash_attention_kernels.forward(cfg, q, k, v)
