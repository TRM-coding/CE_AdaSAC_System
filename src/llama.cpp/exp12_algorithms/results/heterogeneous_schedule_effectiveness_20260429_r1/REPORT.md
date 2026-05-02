# Heterogeneous Load Scheduler Effectiveness

- cgroup root: `/sys/fs/cgroup/ce_ada_llama_6079`
- repeats: `2`
- tokens per decode run: `1`
- baseline: no SVD, same core set and same heterogeneous background load
- model_schedule: model profile + DP scheduler; edge/end offload is skipped because this experiment does not use adb

## Result

| scenario | cores | loads | mode | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |
|---|---:|---|---|---|---|---:|---:|---:|---:|
| `4c_front_hot` | 4 | `90,70,20,0` | `local` | `61,62,63` | `60` | 14 | 3.53011 | 20.4869 | 5.8x |
| `4c_high` | 4 | `70,80,90,100` | `local` | `61,62,63` | `60` | 14 | 21.789 | 18.7242 | 0.859x |
| `4c_mixed` | 4 | `0,30,60,90` | `local` | `60,61,62,63` | `` | 0 | 21.7253 | 21.7253 | 1x |
| `4c_ramp_light` | 4 | `0,10,20,30` | `local` | `60,61,62,63` | `` | 0 | 22.0153 | 22.0153 | 1x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | `62,63,65` | `60,61,64` | 11 | 13.83 | n/a | n/a |
| `6c_high` | 6 | `50,60,70,80,90,100` | `local` | `60,61,62,63,64` | `65` | 14 | 26.8877 | 22.8012 | 0.848x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `local` | `60,61,62,63` | `64,65` | 14 | 23.112 | 23.8429 | 1.03x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `local` | `60,61,62,63,65` | `64` | 14 | 28.7649 | 22.8831 | 0.796x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | `61,62,63,66` | `60,64,65,67` | 1 | 26.4947 | n/a | n/a |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `local` | `61,62,63,64,65,66` | `60,67` | 14 | 30.8872 | 26.2273 | 0.849x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `local` | `60,61,62,63,66,67` | `64,65` | 14 | 28.3043 | 20.266 | 0.716x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `local` | `60,61,62,63,64,65,66` | `67` | 14 | 14.7838 | 0.456795 | 0.0309x |

## Files

- `summary.csv`: aggregated throughput and speedup.
- `schedule_summary.csv`: scheduler choices and estimated latencies.
- `raw.csv`: every decode run with status and output tail.
