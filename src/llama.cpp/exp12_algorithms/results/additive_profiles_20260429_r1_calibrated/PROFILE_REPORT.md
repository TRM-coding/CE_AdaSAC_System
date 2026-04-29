# Additive Scheduler Profiles

- CPUs: `60,61,62,63,64,65,66,67`
- Rates: `0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`
- Scenarios: `load_0, load_10, load_20, load_30, load_40, load_50, load_60, load_70, load_80, load_90, load_100`

| scenario | full local ms | local deadline | request deadline | profile |
|---|---:|---:|---:|---|
| `load_0` | 32.0358 | 33.6376 | 43.2483 | `profile_load_0.json` |
| `load_10` | 35.3874 | 33.6376 | 43.2483 | `profile_load_10.json` |
| `load_20` | 40.4118 | 33.6376 | 43.2483 | `profile_load_20.json` |
| `load_30` | 69.6333 | 33.6376 | 43.2483 | `profile_load_30.json` |
| `load_40` | 38.8865 | 33.6376 | 43.2483 | `profile_load_40.json` |
| `load_50` | 31.8804 | 33.6376 | 43.2483 | `profile_load_50.json` |
| `load_60` | 35.1541 | 33.6376 | 43.2483 | `profile_load_60.json` |
| `load_70` | 31.7830 | 33.6376 | 43.2483 | `profile_load_70.json` |
| `load_80` | 32.4744 | 33.6376 | 43.2483 | `profile_load_80.json` |
| `load_90` | 32.3058 | 33.6376 | 43.2483 | `profile_load_90.json` |
| `load_100` | 37.0624 | 33.6376 | 43.2483 | `profile_load_100.json` |

The scheduler should prefer `p=M`, `rate=0` when that local full-execution plan is feasible, because it has zero loss and the tie-breaker prefers larger `p`.
