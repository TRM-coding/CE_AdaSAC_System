# 20% Load Large Timeout PPL

Settings: `--ctx-size 128 --chunks 1`, load `stress-ng --cpu-load 20`, policy `alternate_0.75`.

| total timeout budget | effective timeout per truncated layer | PPL | delta vs 0ms SVD | eval tok/s |
|---:|---:|---:|---:|---:|
| 0 ms | 0 ms | 69.6940 | +0.0000 | 24.1000 |
| 20 ms | 2 ms | 70.7932 | +1.0992 | 23.3300 |
| 40 ms | 3 ms | 69.9368 | +0.2428 | 21.7100 |
| 60 ms | 5 ms | 69.6940 | +0.0000 | 22.1800 |
| 80 ms | 6 ms | 69.6940 | +0.0000 | 21.3600 |
