# Runtime Optimized Scheduler

- calibration repeats: `2`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `trunc_even_0.9` | 22.0872 | 28.0986 | 1.272x |
| `4c_mixed` | 4 | `trunc_even_0.9` | 22.1838 | 28.3361 | 1.277x |
| `4c_front_hot` | 4 | `trunc_even_0.8` | 20.5616 | 2.69536 | 0.131x |
| `4c_high` | 4 | `trunc_even_0.9` | 22.181 | 28.342 | 1.278x |
| `6c_ramp_light` | 6 | `trunc_even_0.9` | 28.648 | 34.2679 | 1.196x |
| `6c_mixed` | 6 | `baseline_no_svd` | 24.5812 | 23.9103 | 0.973x |
| `6c_front_hot` | 6 | `trunc_even_0.6` | 22.3419 | 0.190691 | 0.009x |
| `6c_high` | 6 | `trunc_even_0.9` | 26.3065 | 37.1623 | 1.413x |
| `8c_ramp_light` | 8 | `trunc_even_0.6` | 4.41521 | 2.40412 | 0.545x |
| `8c_mixed` | 8 | `trunc_even_0.8` | 30.0543 | 27.1752 | 0.904x |
| `8c_front_hot` | 8 | `trunc_even_0.8` | 28.3289 | 30.4723 | 1.076x |
| `8c_high` | 8 | `trunc_even_0.9` | 31.3595 | 41.591 | 1.326x |
