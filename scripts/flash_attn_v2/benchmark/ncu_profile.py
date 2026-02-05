#!/usr/bin/env python3
"""
Profile a single kernel configuration with NVIDIA Nsight Compute.
Output filename is derived from the kernel configuration.

Usage:
    python scripts/flash_attn_v2/benchmark/ncu_profile.py "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles"
    python scripts/flash_attn_v2/benchmark/ncu_profile.py "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles" --seq_len 2048 --d_head 128
    python scripts/flash_attn_v2/benchmark/ncu_profile.py --ref  # Profile flash_attn v2 reference
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

# ===============================
#           local modules
# ===============================
BENCH_SCRIPTS_DIR = pathlib.Path(os.path.dirname(__file__))
SCRIPTS_DIR = BENCH_SCRIPTS_DIR.parent.parent  # scripts/flash_attn_v2/benchmark -> scripts
sys.path.insert(0, str(SCRIPTS_DIR))
from script_utils import setup_project_imports  # noqa: E402

PROJECT_DIR = setup_project_imports(BENCH_SCRIPTS_DIR)

PROFILE_DIR = PROJECT_DIR / "profiles"
RUN_KERNELS_SCRIPT = BENCH_SCRIPTS_DIR / "run_kernels.py"


def sanitize_filename(kernel_name: str) -> str:
    """
    Convert kernel config string to a valid filename.

    Example:
        "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles"
        -> "FP16_128_64_64_4_async_eager_load_0_0_0_tiles"
    """
    # Replace all non-alphanumeric characters (except underscore) with underscore
    name = re.sub(r"[^a-zA-Z0-9_]", "_", kernel_name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile a single kernel with ncu and auto-generate filename from config."
    )
    parser.add_argument(
        "kernel",
        type=str,
        nargs="?",
        default=None,
        help='Kernel config string, e.g., "(FP16, 128, 64, 64, 4): async+eager+load_0_0_0_tiles"',
    )
    parser.add_argument(
        "--ref",
        action="store_true",
        help="Profile flash_attn v2 reference kernel instead",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length (default: 1024)",
    )
    parser.add_argument(
        "--d_head",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
    )
    parser.add_argument(
        "--ncu_set",
        type=str,
        default="full",
        help="ncu --set option (default: full)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory (default: {PROFILE_DIR})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom output filename (without extension). If not provided, derived from kernel config.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the ncu command without executing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.ref and not args.kernel:
        print("Error: Must specify either a kernel config or --ref")
        sys.exit(1)

    # Determine output directory
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else PROFILE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if args.name:
        output_name = args.name
    elif args.ref:
        output_name = f"profile_ref_seq{args.seq_len}_d{args.d_head}"
    else:
        sanitized = sanitize_filename(args.kernel)
        output_name = f"profile_{sanitized}_seq{args.seq_len}_d{args.d_head}"

    output_path = output_dir / output_name

    # Build the run_kernels.py command
    run_cmd = [
        sys.executable,
        str(RUN_KERNELS_SCRIPT),
        str(args.seq_len),
        str(args.d_head),
    ]

    if args.ref:
        run_cmd.append("--ref")
        kernel_regex = "flash_fwd"
    else:
        run_cmd.extend(["--kernels", args.kernel])
        kernel_regex = "flash_forward"

    # Build the ncu command
    ncu_cmd = [
        "ncu",
        f"--set={args.ncu_set}",
        "-k",
        f"regex:{kernel_regex}",
        "-o",
        str(output_path),
    ] + run_cmd

    print(f"Output: {output_path}.ncu-rep")
    print(f"Command: {' '.join(ncu_cmd)}")

    if args.dry_run:
        print("\n(dry run - not executing)")
        return

    print()

    # Run ncu
    result = subprocess.run(ncu_cmd)
    if result.returncode != 0:
        print(f"\nncu exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nProfile saved to: {output_path}.ncu-rep")
    print(f"Open with: ncu-ui {output_path}.ncu-rep")


if __name__ == "__main__":
    main()
