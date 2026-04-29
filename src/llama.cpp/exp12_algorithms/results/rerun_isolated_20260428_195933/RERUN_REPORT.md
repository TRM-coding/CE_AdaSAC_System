# Exp12 Isolated CPU Rerun Report

Date: 2026-04-28

## Environment

- CPU isolation target: 60-79
- Decode CPUs: 60-67
- Load CPUs: 60-63
- Load generator: `stress-ng --cpu 4 --cpu-method matrixprod`
- Model: `src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- Decode binary: `build-release-current/decode_svd_test`
- adb: not used

Before the run, the experiment cgroups were empty and no active `decode_svd_test`/`stress-ng` jobs were found on 60-79. After the run, no experiment jobs were left running.

## Scheduling Effectiveness

Policy compared against baseline: `alternate_0.75`, which applies the adjacent-layer constraint by only truncating odd layers.

| stress-ng load | baseline tok/s | scheduled tok/s | speedup |
|---:|---:|---:|---:|
| 0% | 31.2302 | 34.1198 | 1.093x |
| 20% | 30.5665 | 34.2982 | 1.122x |
| 50% | 12.6369 | 23.6022 | 1.868x |
| 80% | 0.1963 | 0.1685 | 0.858x |
| 100% | 0.1673 | 0.1673 | 1.000x |

At 0-20% load, the schedule gives a stable 9-12% decode speedup. At 50% load, the mean speedup is high, but the raw runs show strong variance because some runs fall to about 3 tok/s while others stay around 30 tok/s. At 80-100% load, the shared load CPUs dominate execution and the scheduling policy cannot recover throughput.

## Timeout Policy

Timeout benchmark uses the same SVD rate policy (`alternate_0.75`) and compares per-layer timeout budget against no-timeout.

| stress-ng load | policy | tok/s | speedup vs no-timeout | text match | top1 match |
|---:|---|---:|---:|---:|---:|
| 20% | timeout_budget_0 | 31.1863 | 1.000x | 1.000 | 1.000 |
| 20% | timeout_budget_2 | 32.4745 | 1.041x | 1.000 | 1.000 |
| 20% | timeout_budget_4 | 18.9769 | 0.608x | 1.000 | 1.000 |
| 20% | timeout_budget_8 | 19.0271 | 0.610x | 1.000 | 1.000 |
| 20% | timeout_budget_16 | 18.3221 | 0.588x | 1.000 | 1.000 |
| 50% | timeout_budget_0 | 32.5598 | 1.000x | 1.000 | 1.000 |
| 50% | timeout_budget_2 | 32.4292 | 0.996x | 1.000 | 1.000 |
| 50% | timeout_budget_4 | 18.0078 | 0.553x | 1.000 | 1.000 |
| 50% | timeout_budget_8 | 17.7406 | 0.545x | 1.000 | 1.000 |
| 50% | timeout_budget_16 | 30.0604 | 0.923x | 1.000 | 1.000 |
| 80% | timeout_budget_0 | 2.3931 | 1.000x | 1.000 | 1.000 |
| 80% | timeout_budget_2 | 0.2994 | 0.125x | 1.000 | 1.000 |
| 80% | timeout_budget_4 | 0.1567 | 0.065x | 1.000 | 1.000 |
| 80% | timeout_budget_8 | 2.1610 | 0.903x | 1.000 | 1.000 |
| 80% | timeout_budget_16 | 0.6498 | 0.272x | 1.000 | 1.000 |

The timeout path preserves the lightweight output proxies in this prompt (`generated_text` and top-1 token both match). Speed-wise, only the 2 ms budget improves at 20% load. At heavier load, timeout budgets do not reliably improve throughput in this local CPU-emulated setup.

## Conclusion

The clean rerun supports the scheduling strategy under light to moderate load: the SVD schedule gives a measurable speedup before the load CPUs become saturated. The timeout strategy is functional and does not change the tested next-token output, but its speed benefit is narrow; the best candidate from this run is `timeout_budget_2` under mild load.

