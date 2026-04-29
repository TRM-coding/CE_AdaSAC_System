# Core Additivity Validation

- CPUs: `60,61,62,63,64,65,66,67`
- Loads: `0,10,20,30,40,50,60,70,80,90,100`
- Repeats: `1`
- Combos: `60,61; 60,61,62,63; 60,61,62,63,64,65; 60,61,62,63,64,65,66,67; 60,62; 61,63`
- Verdict: `direct_split_profile_required`

## Error Summary

| metric | value |
|---|---:|
| count | 66 |
| mean | 0.1941 |
| median | 0.1372 |
| p90 | 0.4012 |
| max | 0.7127 |

## Interpretation

- `additive_ok`: harmonic throughput sum is accurate enough for the scheduler profile.
- `usable_with_calibration`: use the additive model with group-size/load calibration factors.
- `direct_split_profile_required`: do not rely on additivity; profile common core splits directly.
