#!/usr/bin/env python3

import os
import sys

import pathlib

# ===============================
#           local modules
# ===============================

SCRIPTS_DIR = pathlib.Path(os.path.dirname(__file__))
sys.path.append(os.path.abspath(SCRIPTS_DIR.parent))

from toy_attn.flash_attn_v2.kernel_configs import get_kernels_to_build  # noqa: E402


is_first = True


def main():
    with open("./csrc/flash_attn_v2/include/flash_kernels.cuh", "w") as f:
        preamble = """// This file is auto-generated in "gen_kernel_instantiations.py".

#pragma once

#include <map>

#include "flash_attention.cuh"
#include "forward_kernel.cuh"

namespace flash_attn_v2 {

typedef void (*forward_kernel_fn)(const ForwardKernelArgs);

std::map<FlashForwardKernelConfig, forward_kernel_fn>
    forward_kernels = {
"""

        f.write(preamble)

        def add_config(cfg):
            print(cfg.short_form())
            global is_first
            if not is_first:
                f.write(",\n")
            else:
                is_first = False

            cpp_struct_str = cfg.to_cpp_struct()
            f.write(f"        // {cfg.short_form()}\n")
            f.write(f"        \u007b{cpp_struct_str}, &{cfg.kernel_name()}<StaticForwardKernelConfig<{cpp_struct_str}>>\u007d")

        for cfg in get_kernels_to_build():
            add_config(cfg)

        f.write("\n};")
        f.write("\n} // namespace flash")


if __name__ == "__main__":
    main()
