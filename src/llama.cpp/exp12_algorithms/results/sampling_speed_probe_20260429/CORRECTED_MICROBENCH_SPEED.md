# Corrected Sampling-Speed Estimate

The earlier sampling-speed estimate in this directory used `decode_svd_test`.
That was the wrong unit for the planned profile: it measures a full one-token
model forward and also includes model/context setup plus background-load process
setup overhead. It must not be used as the estimate for one-layer or one-matmul
profile sampling.

## Correct Unit

For chapter 5.2 profile construction, the sampling unit should be one layer or
one representative SVD matrix multiplication. The existing microbenchmark
`build-release-current/bench_svd_mul_mat_transformer` is a better starting
point because it benchmarks `ggml_mul_mat_svd` directly on transformer-like
matrix shapes.

## Short Probe

Command:

```bash
sudo -n bash -lc 'CG=/sys/fs/cgroup/ce_ada_llama_6079/run_6067; cd /home/tianruiming/CE_ADA_LLAMA; echo $$ > $CG/cgroup.procs; /usr/bin/time -f "WALL_SECONDS=%e" env LD_LIBRARY_PATH="$PWD/build-release-current/bin" taskset -c 60-67 ./build-release-current/bench_svd_mul_mat_transformer --threads 8 --warmup 2 --repeat 10 --n-cols 1 --type f32'
```

Output:

```text
# threads=8 warmup=2 repeat=10 n_cols=1 type=f32
case,in,out,mid_dim,type,n_cols,fused_median_ms,two_mul_mat_median_ms,speedup_vs_two_mul_mat,fused_avg_ms,two_mul_mat_avg_ms,max_abs_diff
square_512,512,512,512,f32,1,0.120018,0.057978,0.483075,0.177518,0.124859,0.000000
square_1024,1024,1024,1024,f32,1,0.051595,0.086933,1.684923,0.054676,0.092930,0.000000
square_1536,1536,1536,1536,f32,1,0.134479,0.135029,1.004092,0.133449,0.136250,0.000000
qwen_ffn_up_gate_full,1536,8960,1536,f32,1,0.798881,0.833203,1.042964,0.807790,0.836714,0.000000
qwen_ffn_down_full,8960,1536,1536,f32,1,0.893604,0.918120,1.027435,0.902155,0.925375,0.000000
WALL_SECONDS=0.83
```

This single run covers 5 matrix shapes, and for each shape it runs both fused
SVD and two-step matmul variants with 2 warmup + 10 measured iterations. That is
roughly 120 compute calls in 0.83 s, including random tensor construction and
ggml context allocation.

## Corrected Estimate

The previous estimate, about 10.8 s per configuration, was invalid for layer
profile work.

Using the current microbenchmark, the rough cost is closer to:

- about `0.17 s` per matrix shape if each shape is run in a fresh benchmark
  process with the current fused + non-fused comparison code;
- less than that if we benchmark only the fused path needed by the profile;
- lower still if background-load setup is amortized across a batch of Q values
  instead of restarted for every `(q_1, ..., q_8, Q)` point.

Therefore, for profile construction we should not launch `decode_svd_test` per
configuration. We should either extend `bench_svd_mul_mat_transformer` or add a
new dedicated microbenchmark that accepts one matrix size/tail ratio and emits a
single CSV/JSON row.
