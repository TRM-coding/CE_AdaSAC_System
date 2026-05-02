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
| 0 | `baseline_no_svd` | 0 | n/a | n/a | n/a |
| 0 | `model_schedule` | 0 | n/a | n/a | n/a |
| 20 | `baseline_no_svd` | 0 | n/a | n/a | n/a |
| 20 | `model_schedule` | 1 | 21.9883 | 45.4788 | n/a |
| 40 | `baseline_no_svd` | 0 | n/a | n/a | n/a |
| 40 | `model_schedule` | 0 | n/a | n/a | n/a |
| 50 | `baseline_no_svd` | 1 | 25.3928 | 39.3813 | 1x |
| 50 | `model_schedule` | 1 | 25.3928 | 39.3813 | 1x |
| 80 | `baseline_no_svd` | 0 | n/a | n/a | n/a |
| 80 | `model_schedule` | 0 | n/a | n/a | n/a |
| 100 | `baseline_no_svd` | 1 | 19.2018 | 52.0784 | 1x |
| 100 | `model_schedule` | 1 | 20.5862 | 48.5763 | 1.07x |

Schedules with `edge_end` are skipped in actual decode because this rerun intentionally does not use adb or a remote server.
