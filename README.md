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

## Benchmarking

### Run kernels directly

```bash
# Run all kernel configs
python scripts/flash_attn_v2/benchmark/run_kernels.py 1024 128 --kernels all

# Run a specific kernel config (e.g. non-swizzled)
python scripts/flash_attn_v2/benchmark/run_kernels.py 1024 128 --kernels "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles"

# Run a swizzled + double-buffered config
python scripts/flash_attn_v2/benchmark/run_kernels.py 1024 128 --kernels "(BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax"
```

### Profile with NVIDIA Nsight Compute

Profiles are saved to `./profiles/`.

```bash
# Profile all kernels as a table summary
python scripts/flash_attn_v2/benchmark/ncu_bench.py

# Custom d_head and seq_len
python scripts/flash_attn_v2/benchmark/ncu_bench.py --d_heads 128 --seq_lens 1024,2048

# Average over multiple runs
python scripts/flash_attn_v2/benchmark/ncu_bench.py --runs 3
```

### Single kernel ncu profile (for GUI analysis)

```bash
# Profile a single kernel config (auto-generates filename from config)
python scripts/flash_attn_v2/benchmark/ncu_profile.py "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles"
# Output: ./profiles/profile_FP16_128_64_64_4_async_eager_load_0_0_0_tiles_seq1024_d128.ncu-rep

# Custom seq_len and d_head
python scripts/flash_attn_v2/benchmark/ncu_profile.py "(FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles" --seq_len 2048

# Profile reference kernel
python scripts/flash_attn_v2/benchmark/ncu_profile.py --ref

# Dry run (show command without executing)
python scripts/flash_attn_v2/benchmark/ncu_profile.py "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles" --dry_run

# Open in Nsight Compute GUI
ncu-ui ./profiles/profile_FP16_128_64_64_4_async_eager_load_0_0_0_tiles_seq1024_d128.ncu-rep
```

### Compare with flash_attn v2 reference

```bash
# Step 1: Profile the reference kernel (flash_attn v2 package)
python scripts/flash_attn_v2/benchmark/ncu_profile.py --ref

# Step 2: Profile your kernel
python scripts/flash_attn_v2/benchmark/ncu_profile.py "(FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax"

# Step 3: Compare in GUI - open one report, then add the other as baseline
ncu-ui ./profiles/profile_ref_seq1024_d128.ncu-rep
#   → File → Add Baseline → select the other .ncu-rep file
```

Alternatively, profile both in a single run (manual ncu command):

```bash
ncu --set=full -k 'regex:"flash_fwd|flash_forward"' -o ./profiles/profile_compare python scripts/flash_attn_v2/benchmark/run_kernels.py 1024 128 --ref --kernels "(FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax"
```

> **Note (WSL2):** If you see `ERR_NVGPUCTRPERM`, enable GPU performance counters
> on the Windows host: NVIDIA Control Panel → Developer → "Allow access to GPU
> performance counters to all users", then restart WSL2.

## Testing

```bash
# Run all tests
pytest tests/

# Debug double buffer kernel
FA_DEBUG=true pytest tests/test_flash_attention_v2.py::test_debug_double_buffer -s
```

## References

- Flash Attention V1: https://github.com/tspeterkim/flash-attention-minimal
- Flash Attention V2: https://github.com/sonnyli/flash_attention_from_scratch
