# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `trunc_even_0.8` | 21.9912 | 24.7569 | 1.126x |
| `4c_mixed` | 4 | `trunc_even_0.8` | 20.3972 | 22.7851 | 1.117x |
| `4c_front_hot` | 4 | `edge_end` | n/a | n/a | n/a |
| `4c_high` | 4 | `trunc_even_0.8` | 22.3923 | 24.7394 | 1.105x |
| `6c_ramp_light` | 6 | `trunc_even_0.8` | 28.625 | 30.2723 | 1.058x |
| `6c_mixed` | 6 | `edge_end` | n/a | n/a | n/a |
| `6c_front_hot` | 6 | `edge_end` | n/a | n/a | n/a |
| `6c_high` | 6 | `trunc_even_0.8` | 26.9652 | 32.0537 | 1.189x |
| `8c_ramp_light` | 8 | `edge_end` | n/a | n/a | n/a |
| `8c_mixed` | 8 | `edge_end` | n/a | n/a | n/a |
| `8c_front_hot` | 8 | `edge_end` | n/a | n/a | n/a |
| `8c_high` | 8 | `trunc_even_0.8` | 31.1771 | 34.1742 | 1.096x |
