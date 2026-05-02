# Heterogeneous Load Scheduler Effectiveness

- cgroup root: `/sys/fs/cgroup/ce_ada_llama_6079`
- repeats: `1`
- tokens per decode run: `1`
- baseline: no SVD, same core set and same heterogeneous background load
- model_schedule: model profile + DP scheduler; edge/end offload is skipped because this experiment does not use adb

## Result

| scenario | cores | loads | mode | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| `smoke_4c_mixed` | 4 | `0,30,60,90` | `local` | `60,61,62,63` | `` | 0 | 21.6511 | 21.6511 | 1x |

## Files

- `summary.csv`: aggregated throughput and speedup.
- `schedule_summary.csv`: scheduler choices and estimated latencies.
- `raw.csv`: every decode run with status and output tail.
