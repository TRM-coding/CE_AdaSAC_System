# INVALIDATED PPL REPORT

This report used `--ctx-size 32 --chunks 1`; the resulting no-SVD PPL around 287 and scheduled PPL around 2280 are not valid quality measurements. Use `ppl_timeout_small_20260428/PPL_SANITY_CTX128_REPORT.md` instead.

# Timeout PPL Tiny-Sample Experiment

Corpus: `ppl_corpus_qwen_out_64k.txt`; `--ctx-size 32 --chunks 1`.

No-SVD baseline PPL: `287.6802`. This baseline was measured once without load; CPU load should not change dense-model PPL, while timeout can change SVD results because it is wall-clock dependent.

| load | timeout budget | no-SVD PPL | scheduled PPL | delta PPL | PPL ratio | eval tok/s | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0% | 0 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 34.4200 | ok |
| 0% | 2 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 34.2900 | ok |
| 0% | 4 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 34.4300 | ok |
| 0% | 8 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 34.3300 | ok |
| 0% | 16 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 34.2500 | ok |
| 20% | 0 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 22.5700 | ok |
| 20% | 2 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 20.7300 | ok |
| 20% | 4 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 22.0500 | ok |
| 20% | 8 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 21.3300 | ok |
| 20% | 16 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 22.2900 | ok |
| 50% | 0 ms | 287.6802 | 2280.0499 | +1992.3697 | 7.9256x | 4.6100 | ok |
| 50% | 2 ms | 287.6802 | 3738.8240 | +3451.1438 | 12.9965x | 8.7800 | ok |
| 50% | 4 ms | 287.6802 | 2472.5150 | +2184.8348 | 8.5947x | 8.1200 | ok |
| 50% | 8 ms | 287.6802 | 2271.4680 | +1983.7878 | 7.8958x | 9.7400 | ok |
| 50% | 16 ms | 287.6802 | 2499.2155 | +2211.5353 | 8.6875x | 9.8200 | ok |
| 80% | 0 ms | 287.6802 | n/a | n/a | n/a | n/a | timeout >60s |
| 80% | 2 ms | 287.6802 | n/a | n/a | n/a | n/a | timeout >60s |
| 80% | 4 ms | 287.6802 | n/a | n/a | n/a | n/a | timeout >60s |
| 80% | 8 ms | 287.6802 | n/a | n/a | n/a | n/a | timeout >60s |
| 80% | 16 ms | 287.6802 | n/a | n/a | n/a | n/a | timeout >60s |

## Notes

- This is a deliberately tiny PPL sanity check, not a final corpus-level PPL number.
- `ctx=32` makes the absolute PPL noisy and high; use the table mainly to compare timeout-induced direction under the same sample.
- 0% and 20% load show identical PPL across timeout budgets, indicating the timeout did not change the computed result there.
- At 50% load, timeout changes both speed and PPL, so this is the first load level where timeout dropping is visible in this small test.
- At 80% load, the PPL runs did not finish within 60 seconds even with the tiny sample, matching the earlier observation that this load level is too saturated for stable evaluation.
