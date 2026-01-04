from torch.utils.cpp_extension import load


toy_mha_flash_attn = load(
    name="toy_mha_flash_attn",
    sources=[
        "csrc/flash_attn_v1/flash_attn_v1.cu",
        "csrc/flash_attn_v1/binding.cpp",
    ],
    extra_cflags=["-O2", "-std=c++17"],
)

__all__ = ["toy_mha_flash_attn"]
