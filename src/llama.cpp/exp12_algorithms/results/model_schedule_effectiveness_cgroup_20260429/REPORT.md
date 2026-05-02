# Model Schedule Effectiveness Rerun

All decode and load processes were launched through sudo into child cgroups under `/sys/fs/cgroup/ce_ada_llama_6079`.

- run CPUs: `60,61,62,63,64,65,66,67`
- load CPUs: `60,61,62,63,64,65,66,67`
- loads: `0,20,40,50,80,100`
- repeats: `2`
- tokens: `1`

## Summary

| load | policy | runs | tok/s mean | decode ms mean | speedup vs no-SVD |
|---:|---|---:|---:|---:|---:|
| 0 | `baseline_no_svd` | 2 | 28.5959 | 34.9835 | 1x |
| 0 | `model_schedule` | 0 | n/a | n/a | n/a |
| 20 | `baseline_no_svd` | 1 | 30.8666 | 32.3975 | 1x |
| 20 | `model_schedule` | 0 | n/a | n/a | n/a |
| 40 | `baseline_no_svd` | 1 | 30.3722 | 32.9248 | 1x |
| 40 | `model_schedule` | 0 | n/a | n/a | n/a |
| 50 | `baseline_no_svd` | 0 | n/a | n/a | n/a |
| 50 | `model_schedule` | 1 | 25.0492 | 39.9214 | n/a |
| 80 | `baseline_no_svd` | 1 | 24.5519 | 40.73 | 1x |
| 80 | `model_schedule` | 2 | 16.4309 | 61.4149 | 0.669x |
| 100 | `baseline_no_svd` | 2 | 23.6271 | 42.3486 | 1x |
| 100 | `model_schedule` | 0 | n/a | n/a | n/a |

Schedules with `edge_end` are skipped in actual decode because this rerun intentionally does not use adb or a remote server.
