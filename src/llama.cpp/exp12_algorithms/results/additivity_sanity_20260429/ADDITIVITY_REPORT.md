# Core Additivity Validation

- CPUs: `60,61`
- Loads: `0,50,100`
- Repeats: `1`
- Combos: `60,61`
- Verdict: `additive_ok`

## Error Summary

| metric | value |
|---|---:|
| count | 3 |
| mean | 0.0639 |
| median | 0.0632 |
| p90 | 0.0661 |
| max | 0.0661 |

## Interpretation

- `additive_ok`: harmonic throughput sum is accurate enough for the scheduler profile.
- `usable_with_calibration`: use the additive model with group-size/load calibration factors.
- `direct_split_profile_required`: do not rely on additivity; profile common core splits directly.
