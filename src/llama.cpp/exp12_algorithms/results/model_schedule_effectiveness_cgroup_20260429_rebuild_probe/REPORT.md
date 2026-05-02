# Model Schedule Effectiveness Rerun

All decode and load processes were launched through sudo into child cgroups under `/sys/fs/cgroup/ce_ada_llama_6079`.

- run CPUs: `60,61,62,63,64,65,66,67`
- load CPUs: `60,61,62,63,64,65,66,67`
- loads: `0`
- repeats: `2`
- tokens: `1`

## Summary

| load | policy | runs | tok/s mean | decode ms mean | speedup vs no-SVD |
|---:|---|---:|---:|---:|---:|
| 0 | `baseline_no_svd` | 2 | 32.7045 | 30.5776 | 1x |
| 0 | `model_schedule` | 2 | 32.7045 | 30.5776 | 1x |

Schedules with `edge_end` are skipped in actual decode because this rerun intentionally does not use adb or a remote server.
