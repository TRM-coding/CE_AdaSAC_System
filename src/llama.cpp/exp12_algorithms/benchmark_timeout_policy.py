#!/usr/bin/env python3
"""Benchmark per-layer timeout budgets for exp12 local SVD split.

This focuses on the chapter-5 timeout allocation path that can drop the minor
rank slice when it is late.  It compares several total timeout budgets against
a no-timeout reference under the same CPU load.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_effectiveness import parse_list_int, start_load, stop_load
from run_exp12_local import (
    DEFAULT_BINARY,
    DEFAULT_MODEL,
    EXP_DIR,
    cpu_spec,
    make_cgroup,
    parse_cpu_list,
    run_decode,
)
from scheduler import write_rate_file, write_timeout_file


def parse_list_float(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def alternate_rates(n_layers: int, rate: float) -> list[float]:
    return [rate if layer % 2 == 1 else 0.0 for layer in range(n_layers)]


def allocated_timeouts(rates: list[float], total_budget_ms: float) -> list[float]:
    total_weight = sum(rates)
    if total_budget_ms <= 0.0 or total_weight <= 0.0:
        return [0.0] * len(rates)
    return [total_budget_ms * rate / total_weight if rate > 0.0 else 0.0 for rate in rates]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row["cpu_load"], row["policy"]) for row in rows})
    grouped = {key: [] for key in keys}
    for row in rows:
        grouped[(row["cpu_load"], row["policy"])].append(row)

    ref_by_load: dict[int, dict[str, Any]] = {}
    for (cpu_load, policy), items in grouped.items():
        if policy == "timeout_budget_0":
            ref_by_load[cpu_load] = items[0]

    for cpu_load, policy in keys:
        items = grouped[(cpu_load, policy)]
        tok = [float(item["decode_tok_s"]) for item in items if item.get("decode_tok_s")]
        ms = [float(item["generation_decode_ms"]) for item in items if item.get("generation_decode_ms")]
        ref = ref_by_load.get(cpu_load)
        ref_tok = None
        ref_text = None
        ref_top1 = None
        if ref is not None:
            ref_text = ref.get("generated_text")
            ref_top1 = ref.get("top1_id")
            ref_items = grouped.get((cpu_load, "timeout_budget_0"), [])
            ref_vals = [float(item["decode_tok_s"]) for item in ref_items if item.get("decode_tok_s")]
            ref_tok = statistics.fmean(ref_vals) if ref_vals else None

        text_matches = [
            1.0 if item.get("generated_text") == ref_text else 0.0
            for item in items
            if ref_text is not None
        ]
        top1_matches = [
            1.0 if item.get("top1_id") == ref_top1 else 0.0
            for item in items
            if ref_top1 is not None
        ]
        tok_mean = mean(tok)
        out.append(
            {
                "cpu_load": cpu_load,
                "policy": policy,
                "runs": len(tok),
                "decode_tok_s_mean": tok_mean,
                "decode_tok_s_stdev": stdev(tok),
                "generation_decode_ms_mean": mean(ms),
                "speedup_vs_no_timeout": (tok_mean / ref_tok) if tok_mean is not None and ref_tok else None,
                "generated_text_match_rate": mean(text_matches),
                "top1_match_rate": mean(top1_matches),
            }
        )
    return out


def write_summary_md(path: Path, args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp12 Per-Layer Timeout Policy Benchmark",
        "",
        f"日期：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 设置",
        "",
        f"- 模型：`{args.model}`",
        f"- 程序：`{args.binary}`",
        f"- 运行核心：`{args.run_cpus}`",
        f"- 加压核心：`{args.load_cpus}`",
        f"- stress-ng cpu-load：`{args.cpu_loads}`",
        f"- repeats：`{args.repeats}`",
        f"- groupA share：`{args.group_a_share}`",
        f"- SVD rate policy：奇数层 `{args.rate}`，偶数层 `0`",
        "",
        "## 汇总",
        "",
        "| cpu-load | policy | tok/s | speedup | text match | top1 match |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        tok = item["decode_tok_s_mean"]
        speedup = item["speedup_vs_no_timeout"]
        text_match = item["generated_text_match_rate"]
        top1_match = item["top1_match_rate"]
        lines.append(
            "| {cpu_load}% | `{policy}` | {tok} | {speedup} | {text_match} | {top1_match} |".format(
                cpu_load=item["cpu_load"],
                policy=item["policy"],
                tok=f"{tok:.6g}" if tok is not None else "n/a",
                speedup=f"{speedup:.3f}x" if speedup is not None else "n/a",
                text_match=f"{text_match:.3f}" if text_match is not None else "n/a",
                top1_match=f"{top1_match:.3f}" if top1_match is not None else "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `timeout_budget_0` 是 no-timeout 参考：使用同样的 SVD rate 和 A/B split，但不允许 minor slice 超时丢弃。",
            "- 其他策略按论文公式把总 timeout budget 分配到被裁剪层。",
            "- `text match` 和 `top1 match` 是轻量损失代理指标，不等价于完整 perplexity/accuracy。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark exp12 per-layer timeout budgets.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-cpus", default="60-67")
    parser.add_argument("--load-cpus", default="60-63")
    parser.add_argument("--cpu-loads", default="20,50,80")
    parser.add_argument("--load-workers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0.75)
    parser.add_argument("--timeout-budgets-ms", default="0,2,4,8,16")
    parser.add_argument("--group-a-share", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/timeout_policy_latest")
    parser.add_argument("--use-cgroup", action="store_true")
    args = parser.parse_args()

    run_cpus = parse_cpu_list(args.run_cpus)
    load_cpus = parse_cpu_list(args.load_cpus)
    cpu_loads = parse_list_int(args.cpu_loads)
    budgets = parse_list_float(args.timeout_budgets_ms)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_cgroup = make_cgroup("exp12_timeout_run", run_cpus) if args.use_cgroup else None
    load_cgroup = make_cgroup("exp12_timeout_load", load_cpus) if args.use_cgroup else None

    probe = run_decode(args.binary, args.model, run_cpus, args.tokens, cgroup=run_cgroup)
    n_layers = int(probe.get("n_layer") or 28)
    rates = alternate_rates(n_layers, args.rate)
    rates_path = args.out_dir / f"alternate_{args.rate}.rates.txt"
    write_rate_file(rates_path, rates)

    timeout_paths: dict[float, Path] = {}
    for budget in budgets:
        path = args.out_dir / f"timeout_budget_{budget:g}.txt"
        write_timeout_file(path, allocated_timeouts(rates, budget))
        timeout_paths[budget] = path

    group_a = cpu_spec(run_cpus[: len(run_cpus) // 2])
    group_b = cpu_spec(run_cpus[len(run_cpus) // 2 :])
    rows: list[dict[str, Any]] = []

    for cpu_load in cpu_loads:
        for rep in range(args.repeats):
            load = start_load("stress-ng", load_cpus, args.load_workers, cpu_load, load_cgroup)
            try:
                for budget in budgets:
                    result = run_decode(
                        args.binary,
                        args.model,
                        run_cpus,
                        args.tokens,
                        rates=rates_path,
                        group_a=group_a,
                        group_b=group_b,
                        group_a_share=args.group_a_share,
                        minor_timeout_ms=0,
                        svd_offload_timeout_ms=2,
                        layer_timeouts=timeout_paths[budget],
                        cgroup=run_cgroup,
                    )
                    row = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "cpu_load": cpu_load,
                        "repeat": rep,
                        "policy": f"timeout_budget_{budget:g}",
                        "timeout_budget_ms": budget,
                        **result,
                    }
                    rows.append(row)
                    print(
                        f"load={cpu_load} rep={rep} budget={budget:g} "
                        f"tok/s={result.get('decode_tok_s')} text={result.get('generated_text')!r} "
                        f"top1={result.get('top1_id')}"
                    )
            finally:
                stop_load(load)

    summary = summarize(rows)
    (args.out_dir / "raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.out_dir / "summary.csv").open("w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "cpu_load",
                "policy",
                "runs",
                "decode_tok_s_mean",
                "decode_tok_s_stdev",
                "generation_decode_ms_mean",
                "speedup_vs_no_timeout",
                "generated_text_match_rate",
                "top1_match_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    write_summary_md(args.out_dir / "SUMMARY.md", args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
