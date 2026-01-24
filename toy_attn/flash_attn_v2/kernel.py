import os

from torch.utils.cpp_extension import load

if not os.environ.get("MAX_JOBS"):
    os.environ["MAX_JOBS"] = "4"

debug = os.environ.get("FA_DEBUG", "false").lower() == "true"

extra_cuda_cflags = [
    "-std=c++20",
    '-Xcudafe="--diag_suppress=3189"',  # pytorch warnings for c++20
    "--use_fast_math",
    "--generate-line-info",
    "--resource-usage",
    "--expt-relaxed-constexpr",
    "-Xptxas=-warn-lmem-usage",
    "-Xptxas=-warn-spills",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-Xcompiler=-fdiagnostics-color=always",
    "--ftemplate-backtrace-limit=0",
    "--keep",
    "-gencode",
    "arch=compute_80,code=sm_80",
]

if debug:
    extra_cuda_cflags.extend(["-g", "-G", "-DFA_DEBUG", "-O0"])
else:
    extra_cuda_cflags.extend(["-O3"])

flash_attention_kernels = load(
    name="flash_attention_kernels",
    sources=[
        "csrc/flash_attn_v2/flash_attn.cu",
    ],
    extra_cflags=[
        "-O3",
        "-fdiagnostics-color=always",
    ],
    extra_cuda_cflags=extra_cuda_cflags,
    extra_ldflags=[
        "-Wl,--no-as-needed",
        "-lcuda",
    ],
    extra_include_paths=[
        "csrc/flash_attn_v2/include",
    ],
    verbose=True,
)

__all__ = ["flash_attention_kernels"]
