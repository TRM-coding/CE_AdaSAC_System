# Fixed Local Split 6+2, 20% Load

This rerun is after fixing local split semantics: `rate=0.75` assigns the tail rank slice to the minor CPU group and does not drop it unless timeout fires.

Settings:

- run CPUs: `60-67`
- major group: `60-65` (6 cores)
- minor group: `66-67` (2 cores)
- load: `stress-ng --cpu-load 20` on `60-63`
- PPL: `ctx-size=128`, `chunks=1`
- SVD rate policy: `alternate_0.75`

| policy | total timeout | per active layer timeout | PPL | delta PPL | decode tok/s | speedup | top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_no_svd` | - | - | 15.1424 | +0.0000 | 33.2745 | 1.000x | 11 |
| `svd_local_split_6p2` | 0 ms | 0 ms | 15.1424 | +0.0000 | 23.6677 | 0.711x | 304 |
| `svd_local_split_6p2` | 20 ms | 2 ms | 15.2877 | +0.1453 | 23.9636 | 0.720x | 304 |
| `svd_local_split_6p2` | 40 ms | 3 ms | 15.3254 | +0.1830 | 23.6675 | 0.711x | 304 |
| `svd_local_split_6p2` | 60 ms | 5 ms | 15.1424 | +0.0000 | 24.1683 | 0.726x | 304 |
| `svd_local_split_6p2` | 80 ms | 6 ms | 15.1424 | +0.0000 | 23.7638 | 0.714x | 304 |

## Notes

- Baseline and no-timeout local split now have identical PPL (`15.1424`), confirming the tail slice is no longer dropped by default.
- `20 ms` and `40 ms` total budgets show a small PPL increase, which indicates some tail timeout/drop events can occur under 20% load.
- `60 ms` and `80 ms` match baseline PPL on this sample, so these budgets appear wide enough for this load and sample.
- Decode throughput here is single-run and noisy; this table is mainly a correctness/PPL sanity table after the semantic fix.
