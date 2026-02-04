#!/usr/bin/bash

mkdir -p data/logs

CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 FA_DEBUG=true pytest -sv tests/test_flash_attention_v2.py::test_debug_nan_bug_minimal_kernel > data/logs/fa_v2_nan_bug.$(date +%Y-%m-%d-%H%M%S).log 2>&1