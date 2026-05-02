# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `6c_ramp_light` | 6 | `trunc_even_0.85` | 28.3346 | 32.3577 | 1.142x |
| `8c_high` | 8 | `trunc_even_0.85` | 28.935 | 33.0046 | 1.141x |
