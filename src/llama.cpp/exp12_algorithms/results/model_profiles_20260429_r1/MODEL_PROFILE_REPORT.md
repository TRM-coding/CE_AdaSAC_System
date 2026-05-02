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

| scenario | mode | p | clipped | max rate | loss | total ms | full local ms | est. speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| load_0 | local | 8 | 0 | 0.000 | 0.000000 | 32.035794 | 32.035800 | 1.000x |
| load_10 | local | 7 | 14 | 0.400 | 2.329600 | 32.190362 | 35.387400 | 1.099x |
| load_20 | local | 7 | 14 | 0.600 | 3.980100 | 32.175626 | 40.411800 | 1.256x |
| load_30 | edge_end | 4 | 2 | 0.900 | 1.644300 | 33.279494 | 69.633300 | 2.092x |
| load_40 | local | 7 | 14 | 0.900 | 8.495150 | 32.215773 | 51.700301 | 1.605x |
| load_50 | local | 8 | 0 | 0.000 | 0.000000 | 31.880394 | 31.880400 | 1.000x |
| load_60 | local | 7 | 14 | 0.500 | 2.701200 | 32.245332 | 35.154100 | 1.090x |
| load_70 | local | 8 | 0 | 0.000 | 0.000000 | 31.782996 | 31.783000 | 1.000x |
| load_80 | local | 7 | 14 | 0.900 | 11.141100 | 32.200561 | 32.474400 | 1.009x |
| load_90 | local | 7 | 14 | 0.400 | 1.477850 | 32.135378 | 32.305800 | 1.005x |
| load_100 | local | 7 | 14 | 0.600 | 3.980100 | 32.208122 | 37.062400 | 1.151x |
