# Cgroup Rerun Diagnosis

This rerun used sudo-launched child cgroups under:

```text
/sys/fs/cgroup/ce_ada_llama_6079
```

Run and load CPU sets:

```text
run  = 60-67
load = 60-67
```

## Offline Scheduler Result

The model-based scheduler/profile stage completed successfully in the isolated
cgroup.  Profiles and sanity output are in:

```text
src/llama.cpp/exp12_algorithms/results/model_profiles_rerun_cgroup_20260429
```

The scheduler chose:

| load | mode | p | clipped layers | max rate | estimated speedup |
|---:|---|---:|---:|---:|---:|
| 0 | local | 8 | 0 | 0.0 | 1.000x |
| 10 | local | 7 | 14 | 0.4 | 1.099x |
| 20 | local | 7 | 14 | 0.6 | 1.256x |
| 30 | edge_end | 4 | 2 | 0.9 | 2.092x |
| 40 | local | 7 | 14 | 0.9 | 1.605x |
| 50 | local | 8 | 0 | 0.0 | 1.000x |
| 60 | local | 7 | 14 | 0.5 | 1.090x |
| 70 | local | 8 | 0 | 0.0 | 1.000x |
| 80 | local | 7 | 14 | 0.9 | 1.009x |
| 90 | local | 7 | 14 | 0.4 | 1.005x |
| 100 | local | 7 | 14 | 0.6 | 1.151x |

This verifies that the model is wired into the scheduler and that the scheduler
chooses no SVD/no offload when the model predicts full local execution is already
within the deadline.

## Decode Runtime Result

The actual `decode_svd_test` rerun is not a reliable performance result.  Many
runs aborted in the runtime before producing decode throughput:

```text
GGML_ASSERT(buf != NULL && "tensor buffer not set") failed
ggml_backend_tensor_set
llama_context::decode
llama_decode
```

This happened even for a minimal no-SVD baseline launched manually in the
already-existing `run_6067` cgroup:

```text
sudo -n bash -lc 'CG=/sys/fs/cgroup/ce_ada_llama_6079/run_6067; cd /home/tianruiming/CE_ADA_LLAMA; echo $$ > $CG/cgroup.procs; export LD_LIBRARY_PATH=$PWD/build-release-current/bin; taskset -c 60-67 ./build-release-current/decode_svd_test ./src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf 1 8 0 off off off off 0.75 0 2 off'
```

Therefore the failed decode rows should not be interpreted as scheduler
slowdown.  The runtime needs to be fixed or rebuilt before using
`decode_svd_test` for another speed table.

## Valid Decode Rows

The few rows that produced tok/s are preserved in:

```text
summary.csv
raw.csv
```

But coverage is too sparse for a scientific conclusion.  For example:

| load | baseline valid runs | model valid runs | observed speedup |
|---:|---:|---:|---:|
| 50 | 1 | 1 | 1.000x |
| 100 | 1 | 1 | 1.072x |

Other load levels had missing baseline or missing model samples because of the
runtime assertion above.

## Conclusion

The scheduler/model integration rerun succeeded.  The actual decode performance
rerun is inconclusive because the current `decode_svd_test` binary/runtime is
not stable under even a no-SVD baseline in the cgroup.
