# LHS Profile Dataset

This directory contains a measurement-plan dataset for fitting:

```text
Y = f(q_cpu60, q_cpu61, ..., q_cpu67, Q)
```

The dataset combines structured boundary points with Latin hypercube samples.
It is a list of configurations to measure; it does not contain measured latency.

## Metadata

```json
{
  "cpus": [
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67
  ],
  "loads": [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
  ],
  "q_values": [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
  ],
  "lhs_samples_requested": 5000,
  "seed": 20260429,
  "structured_unique_before_lhs": 1375,
  "total_unique": 6375,
  "duplicates_removed": 0,
  "dataset_csv": "src/llama.cpp/exp12_algorithms/results/lhs_profile_dataset_20260429_r1/profile_lhs_dataset.csv"
}
```

## Counts

- total unique configurations: 6375
- requested LHS samples: 5000
- duplicate configurations removed: 0

## Count By Source

| source | count |
|---|---:|
| lhs | 5000 |
| all_equal | 121 |
| major_minor_grid | 110 |
| minor_major_grid | 110 |
| single_core_0 | 110 |
| single_core_1 | 110 |
| single_core_2 | 110 |
| single_core_3 | 110 |
| single_core_4 | 110 |
| single_core_5 | 110 |
| single_core_6 | 110 |
| single_core_7 | 110 |
| prefix_busy | 55 |
| suffix_busy | 55 |
| alternating | 11 |
| gradient | 11 |
| inverse_alternating | 11 |
| inverse_gradient | 11 |

## Count By Q

| Q_pct | count |
|---:|---:|
| 0 | 580 |
| 10 | 579 |
| 20 | 579 |
| 30 | 581 |
| 40 | 579 |
| 50 | 580 |
| 60 | 579 |
| 70 | 579 |
| 80 | 579 |
| 90 | 581 |
| 100 | 579 |

## Notes

- `all_equal`, `single_core_*`, `major_minor_grid`, and shape-pattern rows are deterministic boundary/structure points.
- `lhs` rows provide high-dimensional coverage without enumerating the full grid.
- The CSV should be consumed by a dedicated one-layer / one-matmul microbenchmark, not by the full-model decode benchmark.
