# Fixed Local Split 6+2, 20% Load, rate=0.25

Here `rate=0.25` means the minor group computes the 25% tail rank slice; major group computes the 75% head rank slice. Timeout can drop only the minor tail.

| policy | total timeout | per active layer timeout | PPL | delta PPL | decode tok/s | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_no_svd` | - | - | 15.1424 | +0.0000 | 30.1652 | 1.000x |
| `svd_local_split_6p2_rate025` | 0 ms | 0 ms | 15.1424 | +0.0000 | 26.6076 | 0.882x |
| `svd_local_split_6p2_rate025` | 20 ms | 2 ms | 15.2896 | +0.1472 | 27.5909 | 0.915x |
| `svd_local_split_6p2_rate025` | 40 ms | 3 ms | 15.1529 | +0.0105 | 26.4152 | 0.876x |
| `svd_local_split_6p2_rate025` | 60 ms | 5 ms | 15.0807 | -0.0617 | 27.7179 | 0.919x |
| `svd_local_split_6p2_rate025` | 80 ms | 6 ms | 15.1424 | +0.0000 | 27.5437 | 0.913x |
