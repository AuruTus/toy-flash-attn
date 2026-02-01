# Root Cause Analysis: Double Buffer GEMM Produces Wrong Results

## Summary

The double-buffered kernel produced O_accum values roughly **~1.94x smaller** than the non-buffered kernel, causing the test assertion `diff <= 2 * diff_no_buffer` to fail. The root cause was that the **prefetch lookahead offset** in `matmul()` only checked `DoubleBufferA`, ignoring the case where only B is double-buffered.

## Symptoms

- Double-buffered kernel output diverged significantly from non-buffered baseline
- Max diff from reference was far larger than expected (not a small precision delta)
- The error was systematic: O_accum values were consistently ~half of the correct values

## Debugging Process

### Step 1: Add Debug Prints at Key Stages

Added `#ifdef FA_DEBUG` prints in `forward_kernel.cuh` at each computation stage:

| Location | Variable | Purpose |
|----------|----------|---------|
| Before main loop | Kernel config | Identify DoubleBuffer flags per GEMM |
| After Q@K^T matmul | `S_accum` | Check attention scores |
| After softmax | `m`, `l`, `P_f32` | Check softmax state |
| After P@V matmul | `O_accum` | Check output accumulator |
| After normalization | `O_final` | Check final output |

### Step 2: Compare Intermediate Values

Ran both kernels (double-buffered and non-buffered) on the same input, side-by-side in logs:

| Stage | Double Buffer | Non-Buffer | Match? |
|-------|:---:|:---:|:---:|
| `S_QK_0` (Q@K^T) | `0.0001452 ...` | `0.0001452 ...` | **Identical** |
| `m_0` (row max) | `0.0003862 ...` | `0.0003862 ...` | **Identical** |
| `l_0` (row sum) | `7.9982176 ...` | `7.9982176 ...` | **Identical** |
| `P_f32_0` (softmax) | `0.9996267 ...` | `0.9996267 ...` | **Identical** |
| **`O_accum_0` (P@V)** | **`0.2720129 ...`** | **`0.5280191 ...`** | **DIVERGED** |

The divergence starts exactly at the P@V matmul (`O_PV_GEMM`), while the Q@K^T matmul (`S_QK_GEMM`) was correct.

### Step 3: Identify Why S_QK Works but O_PV Doesn't

Added per-GEMM config prints to show `DoubleBufferA` and `DoubleBufferB`:

**S_QK_GEMM** (A=Q, B=K): `DoubleBufferA=1, DoubleBufferB=1`
Both sides double-buffered — the bug was masked because the lookahead checked `DoubleBufferA` which was true.

**O_PV_GEMM** (A=P, B=V): `DoubleBufferA=0, DoubleBufferB=1`
Only B (V) is double-buffered. P lives entirely in RF (`load_entire_block_into_rf=true`), so `DoubleBufferA=false`.

### Step 4: Trace the Bug in `matmul()`

In `gemm.cuh:111-113`, the prefetch lookahead was:

```cpp
int k_load_fragment =
    k_outer_fragment +
    (GEMM::DoubleBufferA ? GEMM::LoadKTilesPerIter : 0);
//        ^^^^^^^^^^^^^ BUG: only checks A side
```

When `DoubleBufferA=0` (O_PV_GEMM case):
- `k_load_fragment = k_outer_fragment + 0` (no lookahead)
- B's prefetch loads the **same tile** currently being consumed, not the next one
- The MMA then reads from the current stage while the prefetch overwrites it
- Result: half the K-dimension tiles contribute stale/duplicate data

When `DoubleBufferA=1` (S_QK_GEMM case):
- `k_load_fragment = k_outer_fragment + LoadKTilesPerIter` (correct lookahead)
- Works correctly by coincidence

## Root Cause

**The prefetch offset in `matmul()` checked only `DoubleBufferA` instead of `DoubleBuffer` (A or B).** This caused the prefetch to not advance when only B was double-buffered, resulting in the MMA consuming partially overwritten tile data.

## Fix

```cpp
// Before (incorrect):
(GEMM::DoubleBufferA ? GEMM::LoadKTilesPerIter : 0);

// After (correct):
(GEMM::DoubleBuffer ? GEMM::LoadKTilesPerIter : 0);
```

`GEMM::DoubleBuffer` is defined as `DoubleBufferA || DoubleBufferB`, so the lookahead applies whenever either side uses double buffering.

## Lessons Learned

1. **Asymmetric configurations expose hidden assumptions**: The bug only manifested in `O_PV_GEMM` where A (P) is RF-only and B (V) is double-buffered. `S_QK_GEMM` masked the bug because both sides were double-buffered.

2. **Systematic ~2x error suggests skipped computation**: When output values are roughly half of expected, suspect that a loop is processing half the tiles or reprocessing the same tiles twice due to incorrect index arithmetic.

3. **Print per-GEMM config flags**: Logging `DoubleBufferA`/`DoubleBufferB` separately (not just the combined `DoubleBuffer`) immediately revealed the asymmetry.

## Related Files

- `csrc/flash_attn_v2/include/gemm.cuh` - GEMM matmul with double buffer logic (fix applied here)
- `csrc/flash_attn_v2/include/forward_kernel.cuh` - Forward kernel calling both GEMMs
- `csrc/flash_attn_v2/include/static_kernel_config.cuh` - GEMM type definitions (S_QK_GEMM, O_PV_GEMM)
- `csrc/flash_attn_v2/include/debug.cuh` - Debug print utilities
