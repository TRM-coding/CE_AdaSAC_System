# Heterogeneous Load Scheduler Effectiveness

- cgroup root: `/sys/fs/cgroup/ce_ada_llama_6079`
- repeats: `3`
- tokens per decode run: `1`
- baseline: no SVD, same core set and same heterogeneous background load
- model_schedule: model profile + DP scheduler; edge/end offload is skipped because this experiment does not use adb

## Result

The table reports median tok/s over 3 repeats.  Median is used as the primary
metric because per-core `stress-ng --cpu-load` can produce occasional very slow
samples under high contention.

| scenario | cores | loads | mode | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| `4c_front_hot` | 4 | `90,70,20,0` | `local` | `61,62,63` | `60` | 14 | 21.8092 | 20.6436 | 0.947x |
| `4c_high` | 4 | `70,80,90,100` | `local` | `61,62,63` | `60` | 14 | 18.885 | 18.5681 | 0.983x |
| `4c_mixed` | 4 | `0,30,60,90` | `local` | `60,61,62,63` | `` | 0 | 17.9287 | 17.9287 | 1x |
| `4c_ramp_light` | 4 | `0,10,20,30` | `local` | `60,61,62,63` | `` | 0 | 18.182 | 18.182 | 1x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | `62,63,65` | `60,61,64` | 11 | 23.0515 | n/a | n/a |
| `6c_high` | 6 | `50,60,70,80,90,100` | `local` | `60,61,62,63,64` | `65` | 14 | 24.8599 | 22.5555 | 0.907x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `local` | `60,61,62,63` | `64,65` | 14 | 25.6071 | 20.8134 | 0.813x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `local` | `60,61,62,63,65` | `64` | 14 | 25.2776 | 25.1841 | 0.996x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | `61,62,63,66` | `60,64,65,67` | 1 | 28.0908 | n/a | n/a |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `local` | `61,62,63,64,65,66` | `60,67` | 14 | 32.1434 | 25.7346 | 0.801x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `local` | `60,61,62,63,66,67` | `64,65` | 14 | 28.2401 | 23.7564 | 0.841x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `local` | `60,61,62,63,64,65,66` | `67` | 14 | 26.3188 | 20.3892 | 0.775x |

## Interpretation

This run validates the execution path but does not show a positive end-to-end
speedup for the current local split policy.

- In light 4-core cases, the scheduler correctly chooses `p = all cores`,
  `minor = empty`, and `rate = 0`, so performance is exactly the no-SVD
  baseline.
- In the `edge_end` cases, the scheduler predicts that local-only execution is
  not the best choice and selects edge/end offload.  Actual decode is skipped
  for those rows because this experiment intentionally does not use adb.
- In all measured local split cases with nonzero SVD rates, median throughput is
  below the no-SVD baseline.  The best nontrivial median result is
  `4c_high` at `0.983x`; the larger-core split cases are typically `0.80x` to
  `0.91x`.

The main conclusion is therefore negative but useful: the current scheduler is
wired correctly, and it avoids unnecessary scheduling in simple low-load cases,
but the current latency profile/model is too optimistic for real local
major/minor decode.  It underestimates the overhead of splitting the CPU
threadpool, launching the minor SVD tail work, and synchronizing tail results.
Before using this as a final performance claim, the profile needs to measure the
actual local split operator/runtime path, not only standalone matrix latency.

## Files

- `summary.csv`: aggregated throughput and speedup.
- `schedule_summary.csv`: scheduler choices and estimated latencies.
- `raw.csv`: every decode run with status and output tail.
