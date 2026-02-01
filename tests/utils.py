import math
import torch
import einops
from torch.nn import functional as F
from dataclasses import dataclass

from flash_attn import flash_attn_func


def mm_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    p = F.softmax(att, dim=-1)
    return p @ v

def reference_forward_kernel_v2(q, k, v, o=None):
    head_dim = q.shape[-1]
    out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=head_dim**-0.5,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    )
    return out[0]


BATCH_SIZE_FOR_SEQ_LEN = {
    512: 16,
    1024: 16,
    2048: 16,
    4096: 16,
    8192: 8,
    16384: 4,
}
BENCHMARK_N_HEADS = 16


@dataclass(frozen=True)
class QKVConfig:
    n_heads: int
    d_head: int

    batch_size: int
    seq_len: int

    dtype: torch.dtype
    device: torch.device


def generate_qkv(cfg: QKVConfig):
    q = torch.randn(
        (cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    return q, k, v


def generate_qkvo(cfg: QKVConfig):
    all = torch.empty(
        (4, cfg.batch_size, cfg.seq_len, cfg.n_heads, cfg.d_head),
        dtype=cfg.dtype,
        device=cfg.device,
    )
    q, o, k, v = tuple(all[i] for i in range(all.size(0)))
    torch.randn(q.shape, dtype=cfg.dtype, device=cfg.device, out=q)
    torch.randn(k.shape, dtype=cfg.dtype, device=cfg.device, out=k)
    torch.randn(v.shape, dtype=cfg.dtype, device=cfg.device, out=v)
    return q, k, v, o


def py_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    upcast: bool = False,
):
    d_head = q.shape[-1]
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    S = einops.einsum(
        q,
        k,
        "batch seq_q head d_head, batch seq_k head d_head -> batch seq_q head seq_k",
    ) / (d_head**0.5)
    attn_probs = S.softmax(dim=-1)

    out = einops.einsum(
        attn_probs,
        v,
        "batch seq_q head seq_k, batch seq_k head d_head -> batch seq_q head d_head",
    )
    if upcast:
        out = out.to(dtype=dtype_og)
    return out


def error_stats(expected: torch.Tensor, actual: torch.Tensor, atol=1e-5, rtol=1e-3):
    abs_diff = (expected - actual).abs()
    close = torch.isclose(expected, actual, atol=atol, rtol=rtol)
    mismatched = close.numel() - close.sum()
    mismatched_percent = (mismatched / expected.numel()) * 100
    max_diff = abs_diff.max()

    return mismatched, mismatched_percent, max_diff


def evaluate_kernel(cfg, out_ref, out):
    print(f"{cfg.short_form()}")

    mismatched, mismatched_percent, max_diff = error_stats(out_ref, out)
    print(f"  Mismatched elements: {mismatched} / {out.numel()} ({mismatched_percent:.1f}%)")
    print(f"  Greatest absolute difference: {max_diff}")


def get_cuda_device_info(device_idx=0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.cuda.get_device_properties(device_idx)
    return {
        "name": device.name,
        "compute_capability": f"{device.major}.{device.minor}",
        "total_memory": f"{device.total_memory / (1024**3):.2f} GB",
        "multi_processor_count": device.multi_processor_count,
    }


def is_a100():
    return "A100" in get_cuda_device_info()["name"]


def make_debug_tensor(
    batch_size: int,
    seq_len: int,
    n_heads: int,
    d_head: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Generate tensor with values: seq_idx * 0.001 + elem_idx * 0.000001

    e.g., seq 0: 0.001001, 0.001002, ..., 0.001128
          seq 1: 0.002001, 0.002002, ..., 0.002128
    Same values across all n_heads.
    """
    # Create element indices [1, 2, ..., d_head]
    elem_idx = torch.arange(1, d_head + 1, dtype=torch.float32, device=device)
    # Create sequence indices [1, 2, ..., seq_len]
    seq_idx = torch.arange(1, seq_len + 1, dtype=torch.float32, device=device)

    # Compute values: seq_idx * 0.001 + elem_idx * 0.000001
    # Shape: (seq_len, d_head)
    values = seq_idx[:, None] * 0.001 + elem_idx[None, :] * 0.000001

    # Expand to (batch_size, seq_len, n_heads, d_head)
    values = values[None, :, None, :].expand(batch_size, seq_len, n_heads, d_head)
    return values.to(dtype=dtype).contiguous()