# PPL Sanity Check with ctx=128

This replaces the invalid `ctx=32` tiny PPL table. The `ctx=32` result produced no-SVD PPL around 287 and scheduled PPL around 2280, which is not a valid quality conclusion.

Settings: `--ctx-size 128 --chunks 1`, corpus `ppl_corpus_qwen_out_64k.txt`.

| load | policy | timeout budget | PPL | eval tok/s |
|---:|---|---:|---:|---:|
| 0% | `baseline_no_svd` | - | 15.1424 | 32.6500 |
| 0% | `scheduled_timeout` | 0 ms | 69.6940 | 34.2200 |
| 0% | `scheduled_timeout` | 2 ms | 69.6940 | 33.9000 |
| 0% | `scheduled_timeout` | 4 ms | 69.6940 | 33.9300 |
| 0% | `scheduled_timeout` | 8 ms | 69.6940 | 34.0000 |
| 0% | `scheduled_timeout` | 16 ms | 69.6940 | 33.9200 |
| 20% | `baseline_no_svd` | - | 15.1424 | 21.5000 |
| 20% | `scheduled_timeout` | 0 ms | 69.6940 | 21.4000 |
| 20% | `scheduled_timeout` | 2 ms | 82.9594 | 21.7900 |
| 20% | `scheduled_timeout` | 4 ms | 69.8059 | 21.7800 |
| 20% | `scheduled_timeout` | 8 ms | 69.4637 | 22.0700 |
| 20% | `scheduled_timeout` | 16 ms | 69.3648 | 20.9100 |
| 50% | `scheduled_timeout` | 0 ms | 69.6940 | 9.0900 |

## Interpretation

- no-SVD baseline is around `15.1424` on this sample, which is the reasonable reference here.
- SVD+schedule at `alternate_0.75` is around `69.6940` without timeout-triggered changes. That degradation comes from the aggressive SVD truncation policy itself, not from timeout.
- At 0% load, all timeout budgets produce exactly the same PPL, so timeout is not changing the result.
- At 20% load, timeout budget `2 ms` changes PPL to `82.9594`; other budgets stay close to 69. This suggests occasional timeout-driven dropping under load.
- The previous `ctx=32` report should not be used because the context was too short for a meaningful PPL estimate.
