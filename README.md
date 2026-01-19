# Toy Flash Attention

A toy/educational implementation of Flash Attention V1 and V2 with CUDA kernels and PyTorch bindings. This project demonstrates GPU kernel optimization techniques for the attention mechanism, including memory access patterns (swizzling), asynchronous operations, and block-level parallelization.

## Features

- Flash Attention V1 and V2 implementations
- CUDA GPU-accelerated kernels targeting SM_80+ (Ampere and newer)
- PyTorch integration with automatic kernel loading
- Configurable kernel parameters with autotuning support
- Support for FP16 and BF16 data types

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- CUDA Toolkit
- NVIDIA GPU with SM_80+ (Ampere architecture or newer)

## Installation

```bash
pip install -e .
```

## Usage

```python
import torch
from toy_attn.flash_attn_v2.flash_attention import forward
from toy_attn.flash_attn_v2.kernel_configs import FlashForwardKernelConfig

# Create kernel configuration
cfg = FlashForwardKernelConfig()

# Input tensors: (batch, seq_len, n_heads, d_head)
q = torch.randn(1, 512, 8, 128, dtype=torch.float16, device="cuda")
k = torch.randn(1, 512, 8, 128, dtype=torch.float16, device="cuda")
v = torch.randn(1, 512, 8, 128, dtype=torch.float16, device="cuda")

# Run flash attention
output = forward(cfg, q, k, v)
```

## References

- Flash Attention V1: https://github.com/tspeterkim/flash-attention-minimal
- Flash Attention V2: https://github.com/sonnyli/flash_attention_from_scratch
