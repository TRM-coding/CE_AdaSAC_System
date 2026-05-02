# Runtime Optimized Scheduler Final Report

- low/mid-load local policy: interval-layer SVD truncation (`trunc_even_0.8` or tuned `trunc_even_0.85`)
- high-contention policy: `edge_end` (not measured locally because this run intentionally does not use adb)
- local validation metric: median tok/s over 3 repeats under the same heterogeneous background load

## Final Decisions

| scenario | cores | loads | selected policy | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `0,10,20,30` | `trunc_even_0.8` | 21.9912 | 24.7569 | 1.126x |
| `4c_mixed` | 4 | `0,30,60,90` | `trunc_even_0.8` | 20.3972 | 22.7851 | 1.117x |
| `4c_front_hot` | 4 | `90,70,20,0` | `edge_end` | 22.381 | edge_end | edge_end |
| `4c_high` | 4 | `70,80,90,100` | `trunc_even_0.8` | 22.3923 | 24.7394 | 1.105x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `trunc_even_0.85` | 28.3346 | 32.3577 | 1.142x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `edge_end` | 22.7833 | edge_end | edge_end |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | 0.611734 | edge_end | edge_end |
| `6c_high` | 6 | `50,60,70,80,90,100` | `trunc_even_0.8` | 26.9652 | 32.0537 | 1.189x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `edge_end` | 0.451827 | edge_end | edge_end |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `edge_end` | 26.9658 | edge_end | edge_end |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | 27.4939 | edge_end | edge_end |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `trunc_even_0.85` | 28.935 | 33.0046 | 1.141x |

## Summary

- locally validated scenarios: `6`
- local speedup min/median/max: `1.105x / 1.141x / 1.189x`
- local scenarios below baseline: `0`
- local scenarios below 1.10x: `0`
- `edge_end` scenarios are intentionally not scored in this local-only run; they should be validated once the端侧 runtime/server is enabled.
