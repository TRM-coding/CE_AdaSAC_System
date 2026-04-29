# Local Split Drop-Tail Branch Report

## Purpose

This experiment adds an explicit source-code branch for the case where the truncated SVD tail is completely discarded. This is different from the current local split / cooperative path, where the tail rank slice can still be computed by another local CPU group.

The new branch is useful for showing the pure speed effect of SVD truncation: when truncation rate increases, the discarded rank components do not participate in inference.

## Source Changes

- `3dparty/llamacpp/ggml/include/ggml-cpu.h`
  - Added `ggml_cpu_set_svd_force_drop_tail(bool enabled)`.
- `3dparty/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`
  - Added global `g_svd_force_drop_tail`.
  - When enabled, `ggml_mul_mat_svd_use_local_tail_split(...)` returns false, so local split mode no longer computes the tail rank slice.
  - The SVD op then uses the normal `k_keep = total_rank - k_trunc` path.
- `exp6_decode_svd_model/decode_svd_model.cpp`
  - Added optional argument `svd_tail_mode`.
  - Passing `drop_tail`, `drop`, `discard_tail`, or `discard` enables the branch.

## Invocation

The new argument is appended after the existing per-layer timeout argument:

```bash
./build-release-current/decode_svd_test \
  <model> 24 8 0 \
  off <rates_file> \
  60-63 64-67 0.5 0 2 off \
  drop_tail
```

In the log, the branch is visible as:

```text
SVD local split tail mode: drop_tail
```

## Sanity Check

Single-run comparison at truncation rate `0.8`, local split `60-63 + 64-67`, 8 decode threads, 24 tokens:

| mode | decode tok/s | ffn_svd_total |
|---|---:|---:|
| local split default | 23.7142 | 774.691 ms |
| local split + `drop_tail` | 37.0890 | 414.932 ms |

This confirms that the branch changes the compute path: the tail rank slice is no longer computed locally by the second group.

## Full Sweep

Full rerun command:

```bash
python3 rerun_q4_direct_truncation.py \
  --split-a 60-63 \
  --split-b 64-67 \
  --tail-mode drop_tail \
  --rates 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --output-prefix q4_split_drop_tail_truncation_rerun \
  --cgroup-name svd_effect_q4_split_drop_tail_dense
```

Results:

| truncation rate | kept rank | runs | decode tok/s mean | speedup |
|---:|---:|---:|---:|---:|
| 0.0 | 1.0 | 3 | 30.5387 | 1.000x |
| 0.1 | 0.9 | 3 | 31.8524 | 1.043x |
| 0.2 | 0.8 | 3 | 33.0133 | 1.081x |
| 0.3 | 0.7 | 3 | 33.7096 | 1.104x |
| 0.4 | 0.6 | 3 | 35.4694 | 1.161x |
| 0.5 | 0.5 | 3 | 35.8439 | 1.174x |
| 0.6 | 0.4 | 3 | 36.0283 | 1.180x |
| 0.7 | 0.3 | 3 | 36.1345 | 1.183x |
| 0.8 | 0.2 | 3 | 40.4834 | 1.326x |

## Output Files

- `q4_split_drop_tail_truncation_rerun_summary.csv`
- `q4_split_drop_tail_truncation_rerun.pdf`
- `q4_split_drop_tail_truncation_rerun.svg`
- `rerun_q4_direct_truncation.py`
