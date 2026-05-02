# Runtime Policy Model

- training rows: `288`
- best model: `random_forest`
- target: `log(speedup_vs_no_svd)`
- speedup clip for training: `3.000x`
- best grouped-CV speedup MAE: `0.421031`

This model is trained from real `decode_svd_test` runtime measurements, so it captures the observed overhead of local SVD truncation candidates under heterogeneous cgroup load better than the earlier standalone matrix-latency model.
