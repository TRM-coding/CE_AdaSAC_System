# Timeout Profile Diagnosis: major2/minor6, minor load 50%

## Setup

- run cgroup: `/sys/fs/cgroup/ce_ada_llama_6079/run_6067`
- load cgroup: `/sys/fs/cgroup/ce_ada_llama_6079/load_6267`
- run CPUs: `60-67`
- major group: `60-61`
- minor group: `62-67`
- background load: `stress-ng --cpu 6 --cpu-load 50`, pinned to `62-67`
- SVD policy: `alternate_0.25`
- local split semantics: major computes head/main rank, minor computes 25% tail rank
- tokens: 1 decode token

I added runtime counters to `ggml-cpu.c`:

- `*_keep`: minor tail finished before timeout and was kept
- `*_drop`: minor tail missed the timeout and was dropped
- `*_wait`: time spent by the major leader waiting for the minor group after major finished

The new log line is:

```text
[svd-local-timeout-profile] ...
```

## First Diagnostic Sweep

| total timeout | active-layer timeout | decode tok/s | generation decode | timeout profile |
|---:|---:|---:|---:|---|
| 0 ms | 0 ms | 6.6021 | 151.468 ms | no timeout active |
| 20 ms | 2 ms | 22.7830 | 43.892 ms | up/gate/down keep=15/15/15, drop=0/0/0, wait ~= 0.003 ms |
| 40 ms | 3 ms | 27.3549 | 36.557 ms | up/gate/down keep=15/15/15, drop=0/0/0, wait ~= 0.010 ms |
| 60 ms | 5 ms | 1.2204 | 819.374 ms | up/gate/down keep=14/15/15, drop=1/0/0, wait ~= 9.570 ms |
| 80 ms | 6 ms | 25.8801 | 38.640 ms | up/gate/down keep=15/15/15, drop=0/0/0, wait ~= 0.003 ms |

## Extra Repeats

| total timeout | repeat | decode tok/s | generation decode | timeout profile |
|---:|---:|---:|---:|---|
| 20 ms | 0 | 21.9162 | 45.628 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.006 ms |
| 20 ms | 1 | 3.7342 | 267.795 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.652 ms |
| 20 ms | 2 | 26.3347 | 37.973 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.007 ms |
| 60 ms | 0 | 0.9737 | 1027.030 ms | keep=15/14/15, drop=0/1/0, wait ~= 25.859 ms |
| 60 ms | 1 | 9.0924 | 109.982 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.005 ms |
| 60 ms | 2 | 2.5900 | 386.104 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.003 ms |
| 80 ms | 0 | 25.9108 | 38.594 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.003 ms |
| 80 ms | 1 | 8.6383 | 115.763 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.003 ms |
| 80 ms | 2 | 26.7240 | 37.420 ms | keep=15/15/15, drop=0/0/0, wait ~= 0.002 ms |

## Interpretation

The identical no-load PPL values are not evidence about loaded timeout behavior. They only show that when no background load is present, the minor tail is kept and the split computation is numerically equivalent to the no-SVD baseline.

Under minor 50% load, the runtime counters show that minor tail is still almost always kept. In this topology the major group has only 2 cores and computes about 75% of the rank, while the minor group has 6 cores and computes the 25% tail even with 50% synthetic load. Therefore, the minor group is usually not late; the timeout mechanism rarely has a chance to drop anything.

The very slow tok/s points are not explained by timeout wait. Several slow runs have `drop=0` and nearly zero major-wait time. That means the slowdown is coming from CPU scheduling / cgroup load jitter during the actual matmul work, not from "larger timeout waits longer for minor".

The previous mean table should not be interpreted as a stable monotonic relationship between timeout budget and throughput. It was dominated by high variance and a few catastrophic slow runs.

