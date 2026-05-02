# Model-Based Scheduler Profile Report

This profile set connects the trained sklearn latency model to `scheduler.py` by generating the existing `core_splits` JSON schema.

## Important Limitation

The current trained model was fit from fixed-Q measurements. This script therefore uses the model for core-set/load effects and applies linear scaling for layer work and SVD rate. It is a useful scheduler integration, but not yet the final Q-aware Algorithmv2 5.2 profile.

## Generated Profiles

- profiles: 11
- n_layers: 28
- rates: 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
- timeout_budget_ms: 8.0
- idle reference deadline base: 32.035800 ms

## Scheduler Sanity

| scenario | mode | p | clipped | max rate | loss | total ms | full local ms |
|---|---|---:|---:|---:|---:|---:|---:|
| load_0 | local | 7 | 14 | 0.200 | 0.400250 | 31.879717 | 32.035800 |
| load_10 | local | 7 | 14 | 0.500 | 2.517800 | 31.871195 | 35.387400 |
| load_20 | local | 7 | 14 | 0.600 | 4.214600 | 31.855594 | 40.411800 |
| load_30 | none | None | 0 | 0.000 | inf | inf | 69.633300 |
| load_40 | local | 7 | 14 | 0.900 | 8.779150 | 31.894201 | 51.700301 |
| load_50 | local | 7 | 14 | 0.800 | 8.400350 | 31.959007 | 31.880400 |
| load_60 | local | 7 | 14 | 0.500 | 2.895800 | 31.942887 | 35.154100 |
| load_70 | local | 8 | 0 | 0.000 | 0.000000 | 31.782996 | 31.783000 |
| load_80 | local | 7 | 14 | 0.900 | 11.412650 | 31.881426 | 32.474400 |
| load_90 | local | 7 | 14 | 0.400 | 1.666400 | 31.809900 | 32.305800 |
| load_100 | local | 7 | 14 | 0.600 | 4.214600 | 31.887766 | 37.062400 |
