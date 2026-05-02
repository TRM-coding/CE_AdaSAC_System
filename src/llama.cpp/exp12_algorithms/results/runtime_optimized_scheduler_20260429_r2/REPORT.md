# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `trunc_even_0.9` | 22.0059 | 28.4358 | 1.292x |
| `4c_mixed` | 4 | `trunc_even_0.9` | 17.7515 | 21.5304 | 1.213x |
| `4c_front_hot` | 4 | `baseline_no_svd` | 21.4254 | 21.4254 | 1.000x |
| `4c_high` | 4 | `trunc_even_0.9` | 20.201 | 25.9694 | 1.286x |
| `6c_ramp_light` | 6 | `trunc_even_0.9` | 28.0592 | 36.9203 | 1.316x |
| `6c_mixed` | 6 | `baseline_no_svd` | 25.7929 | 25.7929 | 1.000x |
| `6c_front_hot` | 6 | `baseline_no_svd` | 23.2345 | 23.2345 | 1.000x |
| `6c_high` | 6 | `trunc_even_0.9` | 25.4855 | 36.7719 | 1.443x |
| `8c_ramp_light` | 8 | `baseline_no_svd` | 32.038 | 32.038 | 1.000x |
| `8c_mixed` | 8 | `baseline_no_svd` | 24.4155 | 24.4155 | 1.000x |
| `8c_front_hot` | 8 | `baseline_no_svd` | 25.1563 | 25.1563 | 1.000x |
| `8c_high` | 8 | `trunc_even_0.9` | 26.6859 | 41.4828 | 1.554x |
