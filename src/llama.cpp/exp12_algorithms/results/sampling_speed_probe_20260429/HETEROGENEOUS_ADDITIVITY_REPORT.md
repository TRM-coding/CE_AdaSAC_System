# Heterogeneous Core Additivity Validation

- Single-core source: `src/llama.cpp/exp12_algorithms/results/additivity_full_20260429_r1/single_core.csv`
- Repeats: `1`

| name | cpus | load vector | measured ms | predicted ms | relative error |
|---|---|---|---:|---:|---:|
| `probe_2_easy` | `60,61` | `60:0,61:50` | 85.8467 | 78.6581 | 0.0837 |
| `probe_4_balanced` | `60,61,62,63` | `60:0,61:0,62:50,63:50` | 47.2585 | 39.3762 | 0.1668 |
| `probe_4_bad` | `60,61,62,63` | `60:0,61:20,62:60,63:100` | 131.9340 | 39.8240 | 0.6982 |
| `probe_8_gradient` | `60,61,62,63,64,65,66,67` | `60:0,61:10,62:20,63:30,64:60,65:70,66:90,67:100` | 38.0266 | 19.9499 | 0.4754 |
| `probe_8_bad` | `60,61,62,63,64,65,66,67` | `60:0,61:0,62:0,63:0,64:80,65:80,66:100,67:100` | 5998.6300 | 19.7901 | 0.9967 |

## Summary

```json
{
  "count": 5,
  "mean": 0.4841502282976348,
  "median": 0.4753700214427341,
  "max": 0.9967008958077453
}
```
