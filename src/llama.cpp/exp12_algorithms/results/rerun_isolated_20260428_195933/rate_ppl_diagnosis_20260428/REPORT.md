# SVD Rate PPL Diagnosis

Settings: `ctx-size=128`, `chunks=1`, no extra load. The rate is applied only to odd layers (`alternate_rate`), matching the previous scheduling experiments.

| rate on odd layers | kept rank fraction | PPL | eval tok/s |
|---:|---:|---:|---:|
| 0.00 | 1.00 | 15.1424 | 32.8000 |
| 0.10 | 0.90 | 31.8625 | 30.6100 |
| 0.25 | 0.75 | 38.3265 | 31.7400 |
| 0.50 | 0.50 | 60.5598 | 33.2800 |
| 0.75 | 0.25 | 69.6940 | 34.2800 |

Important: runtime interprets `rate` as truncated/offloaded tail rank. In local no-server mode this becomes rank truncation. Therefore `rate=0.75` drops 75% of rank on selected layers and keeps only 25%, which is very aggressive.
