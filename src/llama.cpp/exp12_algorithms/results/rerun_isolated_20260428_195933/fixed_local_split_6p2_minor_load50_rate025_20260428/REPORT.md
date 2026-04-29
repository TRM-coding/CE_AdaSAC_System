# Fixed Local Split 6+2, Minor Group 50% Load

Experiment: load is on the minor CPU group only.

- run CPUs: `60-67`
- major group: `60-65` (6 cores)
- minor group: `66-67` (2 cores)
- background load: `stress-ng --cpu 2 --cpu-load 50` on `66-67`
- SVD rate policy: `alternate_0.25`; minor group computes the 25% tail rank slice on odd layers
- repeats: 3

| policy | total timeout | per active layer timeout | tok/s mean | stdev | decode ms mean | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_no_svd` | - | - | 31.4059 | 2.0829 | 31.9376 | 1.000x |
| `svd_local_split_6p2_rate025` | 0 ms | 0 ms | 28.1511 | 0.1469 | 35.5232 | 0.896x |
| `svd_local_split_6p2_rate025` | 20 ms | 2 ms | 27.0386 | 0.5275 | 36.9937 | 0.861x |
| `svd_local_split_6p2_rate025` | 40 ms | 3 ms | 18.3195 | 12.0963 | 99.7927 | 0.583x |
| `svd_local_split_6p2_rate025` | 60 ms | 5 ms | 27.7503 | 0.2774 | 36.0381 | 0.884x |
| `svd_local_split_6p2_rate025` | 80 ms | 6 ms | 26.0304 | 2.6268 | 38.6952 | 0.829x |

## Raw Runs

| repeat | policy | total timeout | tok/s | decode ms | generated text | top1 |
|---:|---|---:|---:|---:|---|---:|
| 0 | `baseline_no_svd` | - | 29.0707 | 34.3988 | `,` | 11 |
| 0 | `svd_local_split_6p2_rate025` | 0 ms | 28.0035 | 35.7098 | `,` | 11 |
| 0 | `svd_local_split_6p2_rate025` | 20 ms | 27.4056 | 36.4889 | `,` | 11 |
| 0 | `svd_local_split_6p2_rate025` | 40 ms | 27.2626 | 36.6803 | `,` | 11 |
| 0 | `svd_local_split_6p2_rate025` | 60 ms | 27.9682 | 35.7548 | `,` | 11 |
| 0 | `svd_local_split_6p2_rate025` | 80 ms | 27.7179 | 36.0778 | `,` | 11 |
| 1 | `baseline_no_svd` | - | 33.0722 | 30.2369 | `,` | 11 |
| 1 | `svd_local_split_6p2_rate025` | 0 ms | 28.1526 | 35.5207 | `,` | 11 |
| 1 | `svd_local_split_6p2_rate025` | 20 ms | 26.4341 | 37.8300 | `,` | 11 |
| 1 | `svd_local_split_6p2_rate025` | 40 ms | 23.1397 | 43.2157 | `,` | 11 |
| 1 | `svd_local_split_6p2_rate025` | 60 ms | 27.4380 | 36.4459 | `,` | 11 |
| 1 | `svd_local_split_6p2_rate025` | 80 ms | 27.3694 | 36.5371 | `,` | 11 |
| 2 | `baseline_no_svd` | - | 32.0747 | 31.1772 | `,` | 11 |
| 2 | `svd_local_split_6p2_rate025` | 0 ms | 28.2973 | 35.3391 | `,` | 11 |
| 2 | `svd_local_split_6p2_rate025` | 20 ms | 27.2761 | 36.6621 | `,` | 11 |
| 2 | `svd_local_split_6p2_rate025` | 40 ms | 4.5562 | 219.4820 | `,` | 11 |
| 2 | `svd_local_split_6p2_rate025` | 60 ms | 27.8447 | 35.9135 | `,` | 11 |
| 2 | `svd_local_split_6p2_rate025` | 80 ms | 23.0040 | 43.4707 | `,` | 11 |
