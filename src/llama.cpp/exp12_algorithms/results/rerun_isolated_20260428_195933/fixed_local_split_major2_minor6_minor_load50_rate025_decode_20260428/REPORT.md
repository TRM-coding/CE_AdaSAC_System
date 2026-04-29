# Major 2 / Minor 6 with Minor 50% Load: Decode Performance

- run CPUs: `60-67`
- major group: `60-61` (2 cores, no background load)
- minor group: `62-67` (6 cores)
- background load: `stress-ng --cpu 6 --cpu-load 50` on `62-67`
- SVD rate policy: `alternate_0.25`; minor computes 25% tail rank on odd layers
- repeats: 3
- Note: loaded PPL with `ctx=128` and `ctx=64` timed out in this topology; PPL data must be run separately with a lighter setup or longer wall time.

| policy | total timeout | per active layer timeout | tok/s mean | stdev | decode ms mean | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_no_svd` | - | - | 11.9502 | 18.4270 | 2158.3250 | 1.000x |
| `svd_local_split_major2_minor6_rate025` | 0 ms | 0 ms | 27.3206 | 1.2995 | 36.6592 | 2.286x |
| `svd_local_split_major2_minor6_rate025` | 20 ms | 2 ms | 7.7918 | 12.5011 | 1719.0402 | 0.652x |
| `svd_local_split_major2_minor6_rate025` | 40 ms | 3 ms | 17.8540 | 15.1314 | 779.1868 | 1.494x |
| `svd_local_split_major2_minor6_rate025` | 60 ms | 5 ms | 8.9258 | 11.8344 | 2160.3918 | 0.747x |
| `svd_local_split_major2_minor6_rate025` | 80 ms | 6 ms | 1.5635 | 2.4265 | 4176.8533 | 0.131x |

## Raw Runs

| repeat | policy | timeout | tok/s | decode ms | generated text | top1 |
|---:|---|---:|---:|---:|---|---:|
| 0 | `baseline_no_svd` | - | 2.4999 | 400.0110 | `,` | 11 |
| 0 | `svd_local_split_major2_minor6_rate025` | 0 ms | 27.8989 | 35.8437 | `,` | 11 |
| 0 | `svd_local_split_major2_minor6_rate025` | 20 ms | 22.2219 | 45.0006 | `,` | 11 |
| 0 | `svd_local_split_major2_minor6_rate025` | 40 ms | 27.8164 | 35.9500 | `,` | 11 |
| 0 | `svd_local_split_major2_minor6_rate025` | 60 ms | 0.1613 | 6200.0100 | `,` | 11 |
| 0 | `svd_local_split_major2_minor6_rate025` | 80 ms | 4.3655 | 229.0700 | `,` | 11 |
| 1 | `baseline_no_svd` | - | 33.1851 | 30.1340 | `,` | 11 |
| 1 | `svd_local_split_major2_minor6_rate025` | 0 ms | 25.8323 | 38.7113 | `,` | 11 |
| 1 | `svd_local_split_major2_minor6_rate025` | 20 ms | 0.9038 | 1106.3900 | `,` | 11 |
| 1 | `svd_local_split_major2_minor6_rate025` | 40 ms | 25.3034 | 39.5203 | `,` | 11 |
| 1 | `svd_local_split_major2_minor6_rate025` | 60 ms | 4.2284 | 236.4980 | `,` | 11 |
| 1 | `svd_local_split_major2_minor6_rate025` | 80 ms | 0.1622 | 6166.6700 | `,` | 11 |
| 2 | `baseline_no_svd` | - | 0.1654 | 6044.8300 | `,` | 11 |
| 2 | `svd_local_split_major2_minor6_rate025` | 0 ms | 28.2306 | 35.4225 | `,` | 11 |
| 2 | `svd_local_split_major2_minor6_rate025` | 20 ms | 0.2496 | 4005.7300 | `,` | 11 |
| 2 | `svd_local_split_major2_minor6_rate025` | 40 ms | 0.4421 | 2262.0900 | `,` | 11 |
| 2 | `svd_local_split_major2_minor6_rate025` | 60 ms | 22.3877 | 44.6674 | `,` | 11 |
| 2 | `svd_local_split_major2_minor6_rate025` | 80 ms | 0.1630 | 6134.8200 | `,` | 11 |

## PPL Reference Without Background Load

Loaded PPL with `ctx=128` and `ctx=64` timed out for this topology. The table below is a no-background-load quality reference for the same major/minor split and timeout files; it verifies the split itself does not alter PPL when timeout pressure is absent.

| timeout | PPL | eval tok/s |
|---:|---:|---:|
| baseline | 15.1424 | 32.2200 |
| 0 ms | 15.1424 | 27.4700 |
| 20 ms | 15.1424 | 26.7800 |
| 40 ms | 15.1424 | 26.9600 |
| 60 ms | 15.1424 | 26.9600 |
| 80 ms | 15.1424 | 26.7500 |
