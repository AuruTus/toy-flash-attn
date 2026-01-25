# Root Cause Analysis: NaN in Flash Attention Softmax

## Summary

The output tensor contained all NaN values due to **swapped arguments** in the `calc_row_max` function call, causing uninitialized memory to be used in the softmax computation.

## Symptoms

- Output SMEM matrix (`O`) contained all NaN values
- `l` (row sum accumulator) became NaN after `scale_l_O` on iteration 0
- `m_next` values were different across warp lanes (should be identical after reduction)

## Debugging Process

### Step 1: Add Debug Prints

Added comprehensive debug prints at key synchronization points:

| Location | Variable | Purpose |
|----------|----------|---------|
| After Q SM2RF | `Q.data()` | Verify Q loaded correctly |
| After K SM2RF | `K.data()` | Verify K loaded correctly |
| After QK matmul | `S_accum` | Check attention scores |
| After `calc_row_max` | `m_next` | Check row max values |
| After `scale_l_O` | `l`, `O_accum` | Check scaling step |
| After exponentiate | `S_accum` | Check exp values |
| After PV matmul | `O_accum` | Check output accumulator |

### Step 2: Trace NaN Origin

From logs, observed the NaN propagation:

```
l_before_scale_0: 0.0 (correct initial value)
l_after_scale_0:  nan (BUG INTRODUCED HERE)
```

### Step 3: Analyze `scale_l_O` Function

```cpp
scale = expf(m_curr[q] - m_next[q]);
l[q] *= scale;
```

NaN occurs when `scale = expf(NaN)`, which happens when:
- `m_curr - m_next = -inf - (-inf) = NaN`, OR
- Either operand is already NaN

### Step 4: Examine `m_next` Values

From debug output:
```
m_next_0 REGS:
0.0000000 0.0052367 0.0054419 0.0056470   <- lanes 0,1,2,3
0.0055868 0.0058145 0.0060423 0.0062700   <- lanes 4,5,6,7
```

**Critical observation:** Each lane has a DIFFERENT value. After warp reduction, all lanes within a row group (0-3, 4-7, etc.) should have the SAME max value.

### Step 5: Check `calc_row_max` Call Site

**Function signature:**
```cpp
void calc_row_max(
    accum_t (&S_accum)[...],
    accum_t (&m_next)[...],   // OUTPUT - new max
    accum_t (&m_curr)[...]    // INPUT  - current max
)
```

**Incorrect call (before fix):**
```cpp
calc_row_max(S_accum.data(), m, m_next);
//                           ^      ^
//                       OUTPUT  INPUT  (SWAPPED!)
```

**What happened:**
1. `m` (initialized to `-inf`) passed as output parameter
2. `m_next` (uninitialized local array) passed as input parameter
3. Function copies uninitialized garbage: `m_next[q] = m_curr[q]`
4. Lane 0 happened to have `0.0` in uninitialized memory
5. Warp reduction didn't help because initial values were wrong

## Root Cause

**Swapped arguments in `calc_row_max` call.**

The function expected `(S_accum, output, input)` but received `(S_accum, input, output)`.

## Fix

```cpp
// Before (incorrect):
calc_row_max(S_accum.data(), m, m_next);

// After (correct):
calc_row_max(S_accum.data(), m_next, m);
```

## Lessons Learned

1. **Parameter naming matters**: The function parameters `m_next` and `m_curr` were confusing. Consider renaming to `m_out` and `m_in` for clarity.

2. **Debug predictable data**: Using the debug tensor generator with values like `0.001001, 0.001002, ...` made it easier to trace data flow.

3. **Check intermediate values**: Adding RF (register file) debug prints at each computation step quickly isolated where NaN was introduced.

4. **Warp reduction sanity check**: After any warp shuffle reduction, all participating lanes should have identical values. Different values indicate a bug.

## Related Files

- `csrc/flash_attn_v2/include/softmax.cuh` - Softmax functions
- `csrc/flash_attn_v2/include/forward_kernel.cuh` - Forward kernel with fix
- `csrc/flash_attn_v2/include/debug.cuh` - Debug print utilities
