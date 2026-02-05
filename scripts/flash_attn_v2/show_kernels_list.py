#!/usr/bin/env python3

import os
import sys

import pathlib

# ===============================
#           local modules
# ===============================
SCRIPTS_DIR = pathlib.Path(os.path.dirname(__file__))
sys.path.insert(0, str(SCRIPTS_DIR.parent))
from script_utils import setup_project_imports  # noqa: E402

setup_project_imports(SCRIPTS_DIR)

from toy_attn.flash_attn_v2.kernel_configs import (  # noqa: E402
    get_kernels_to_build,
)


def main():
    raw_kernel_configs = get_kernels_to_build()
    for cfg in raw_kernel_configs:
        print(cfg.short_form())


if __name__ == "__main__":
    main()
