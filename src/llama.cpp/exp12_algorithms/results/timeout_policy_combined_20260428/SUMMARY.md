# Exp12 Per-Layer Timeout Policy Combined Summary

| cpu-load | policy | runs | tok/s mean | tok/s median | speedup mean | speedup median | text match | top1 match |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | `timeout_budget_0` | 3 | 31.7458 | 31.7275 | 1.000x | 1.000x | 1.000 | 1.000 |
| 20% | `timeout_budget_16` | 3 | 23.0352 | 31.8556 | 0.726x | 1.004x | 1.000 | 1.000 |
| 20% | `timeout_budget_2` | 3 | 32.3662 | 32.4727 | 1.020x | 1.023x | 1.000 | 1.000 |
| 20% | `timeout_budget_8` | 3 | 31.9154 | 32.3083 | 1.005x | 1.018x | 1.000 | 1.000 |
| 50% | `timeout_budget_0` | 3 | 31.8236 | 32.1916 | 1.000x | 1.000x | 1.000 | 1.000 |
| 50% | `timeout_budget_16` | 3 | 21.9296 | 30.4109 | 0.689x | 0.945x | 1.000 | 1.000 |
| 50% | `timeout_budget_2` | 3 | 31.7082 | 31.6183 | 0.996x | 0.982x | 1.000 | 1.000 |
| 50% | `timeout_budget_8` | 3 | 31.8354 | 31.7372 | 1.000x | 0.986x | 1.000 | 1.000 |
| 80% | `timeout_budget_0` | 2 | 0.15728 | 0.15728 | 1.000x | 1.000x | 1.000 | 1.000 |
| 80% | `timeout_budget_2` | 2 | 0.153362 | 0.153362 | 0.975x | 0.975x | 1.000 | 1.000 |
| 80% | `timeout_budget_8` | 2 | 0.152323 | 0.152323 | 0.968x | 0.968x | 1.000 | 1.000 |
