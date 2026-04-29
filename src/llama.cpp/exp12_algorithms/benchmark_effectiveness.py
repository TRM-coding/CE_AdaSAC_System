#!/usr/bin/env python3
"""Benchmark whether exp12 SVD rate schedules help under CPU load.

The benchmark is intentionally local-only and does not use adb.  It keeps a
busy-loop load on selected CPUs while running several SVD truncation policies
through the existing `decode_svd_test` entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from run_exp12_local import (
    DEFAULT_BINARY,
    DEFAULT_MODEL,
    EXP_DIR,
    cpu_spec,
    make_cgroup,
    parse_cpu_list,
    run_decode,
)
from scheduler import write_rate_file


def parse_list_int(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def start_load(
    mode: str,
    cpus: list[int],
    workers: int,
    cpu_load: int,
    cgroup: Path | None,
) -> list[subprocess.Popen[str]]:
    if workers <= 0:
        return []
    cpu_range = cpu_spec(cpus)
    if mode == "busy":
        commands = [["taskset", "-c", str(cpu), "bash", "-lc", "while :; do :; done"] for cpu in cpus[:workers]]
    elif mode == "stress-ng":
        commands = [[
            "taskset",
            "-c",
            cpu_range,
            "stress-ng",
            "--cpu",
            str(workers),
            "--cpu-load",
            str(cpu_load),
            "--cpu-method",
            "matrixprod",
            "--quiet",
        ]]
    else:
        raise ValueError(f"unknown load mode: {mode}")

    procs: list[subprocess.Popen[str]] = []
    for cmd in commands:
        if cgroup is not None:
            script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
            cmd = ["sudo", "-n", "bash", "-lc", script]
        procs.append(
            subprocess.Popen(
                cmd,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
    time.sleep(0.8)
    return procs


def stop_load(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def make_rates(n_layers: int, policy: str) -> list[float] | None:
    if policy == "baseline":
        return None
    if policy.startswith("uniform_"):
        return [float(policy.split("_", 1)[1])] * n_layers
    if policy.startswith("alternate_"):
        rate = float(policy.split("_", 1)[1])
        return [rate if layer % 2 == 1 else 0.0 for layer in range(n_layers)]
    raise ValueError(f"unknown policy: {policy}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row["load_workers"], row["policy"]) for row in rows})
    by_key = {(load, policy): [] for load, policy in keys}
    for row in rows:
        by_key[(row["load_workers"], row["policy"])].append(row)

    baseline_by_load: dict[int, float] = {}
    for (load, policy), items in by_key.items():
        vals = [float(item["decode_tok_s"]) for item in items if item.get("decode_tok_s")]
        if policy == "baseline" and vals:
            baseline_by_load[load] = statistics.fmean(vals)

    for load, policy in keys:
        vals = [float(item["decode_tok_s"]) for item in by_key[(load, policy)] if item.get("decode_tok_s")]
        ms_vals = [float(item["generation_decode_ms"]) for item in by_key[(load, policy)] if item.get("generation_decode_ms")]
        avg = mean(vals)
        base = baseline_by_load.get(load)
        speedup = (avg / base) if avg is not None and base else None
        out.append(
            {
                "load_workers": load,
                "policy": policy,
                "runs": len(vals),
                "decode_tok_s_mean": avg,
                "decode_tok_s_stdev": stdev(vals),
                "generation_decode_ms_mean": mean(ms_vals),
                "speedup_vs_same_load_baseline": speedup,
            }
        )
    return out


def write_summary_md(path: Path, args: argparse.Namespace, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp12 调度有效性实验",
        "",
        f"日期：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验目的",
        "",
        "验证在电脑端 CPU 存在额外负载时，SVD 截断率调度是否能提升 decode 速度。",
        "本实验不使用 adb，不验证真实手机后缀卸载；这里只验证当前可落地的本地 SVD rate 调度。",
        "",
        "## 设置",
        "",
        f"- 模型：`{args.model}`",
        f"- 程序：`{args.binary}`",
        f"- 运行核心：`{args.run_cpus}`",
        f"- 加压核心：`{args.load_cpus}`",
        f"- 负载模式：`{args.load_mode}`",
        f"- stress-ng cpu-load：`{args.cpu_load}`",
        f"- tokens：`{args.tokens}`",
        f"- repeats：`{args.repeats}`",
        f"- policies：`{args.policies}`",
        f"- cgroup：`{'enabled' if args.use_cgroup else 'disabled'}`",
        "",
        "## 汇总",
        "",
        "| load workers | policy | runs | tok/s mean | decode ms mean | speedup |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        tok = item["decode_tok_s_mean"]
        ms = item["generation_decode_ms_mean"]
        speedup = item["speedup_vs_same_load_baseline"]
        lines.append(
            "| {load_workers} | `{policy}` | {runs} | {tok} | {ms} | {speedup} |".format(
                load_workers=item["load_workers"],
                policy=item["policy"],
                runs=item["runs"],
                tok=f"{tok:.6g}" if tok is not None else "n/a",
                ms=f"{ms:.6g}" if ms is not None else "n/a",
                speedup=f"{speedup:.3f}x" if speedup is not None else "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## 判定标准",
            "",
            "- 同一 `load_workers` 下，`speedup > 1.0x` 表示该策略相对同负载 baseline 提速。",
            "- `alternate_*` 策略模拟第五章相邻层不可同时裁剪约束。",
            "- `uniform_*` 策略作为激进截断对照，速度可能更快，但精度风险也更高。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark exp12 schedule effectiveness under CPU load.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-cpus", default="60-67")
    parser.add_argument("--load-cpus", default="60-63")
    parser.add_argument("--load-workers-list", default="0,4")
    parser.add_argument("--load-mode", choices=["stress-ng", "busy"], default="stress-ng")
    parser.add_argument("--cpu-load", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--policies",
        default="baseline,alternate_0.5,alternate_0.75,uniform_0.5",
        help="Comma-separated policies: baseline, alternate_RATE, uniform_RATE",
    )
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/effectiveness_latest")
    parser.add_argument("--use-cgroup", action="store_true")
    args = parser.parse_args()

    run_cpus = parse_cpu_list(args.run_cpus)
    load_cpus = parse_cpu_list(args.load_cpus)
    load_workers_list = parse_list_int(args.load_workers_list)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_cgroup = make_cgroup("exp12_effectiveness_run", run_cpus) if args.use_cgroup else None
    load_cgroup = make_cgroup("exp12_effectiveness_load", load_cpus) if args.use_cgroup else None

    # Probe once to learn layer count.  This is also a useful no-load baseline,
    # but it is repeated below so every policy follows the same loop shape.
    probe = run_decode(args.binary, args.model, run_cpus, args.tokens, cgroup=run_cgroup)
    n_layers = int(probe.get("n_layer") or 28)

    rate_paths: dict[str, Path | str] = {"baseline": "off"}
    for policy in policies:
        rates = make_rates(n_layers, policy)
        if rates is None:
            rate_paths[policy] = "off"
            continue
        path = args.out_dir / f"{policy}.rates.txt"
        write_rate_file(path, rates)
        rate_paths[policy] = path

    group_a = cpu_spec(run_cpus[: len(run_cpus) // 2])
    group_b = cpu_spec(run_cpus[len(run_cpus) // 2 :])
    rows: list[dict[str, Any]] = []

    for load_workers in load_workers_list:
        for rep in range(args.repeats):
            busy = start_load(args.load_mode, load_cpus, load_workers, args.cpu_load, load_cgroup)
            try:
                for policy in policies:
                    result = run_decode(
                        args.binary,
                        args.model,
                        run_cpus,
                        args.tokens,
                        rates=rate_paths[policy],
                        group_a=group_a if policy != "baseline" else "off",
                        group_b=group_b if policy != "baseline" else "off",
                        group_a_share=0.5,
                        cgroup=run_cgroup,
                    )
                    row = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "load_workers": load_workers,
                        "repeat": rep,
                        "policy": policy,
                        **result,
                    }
                    rows.append(row)
                    print(
                        f"load={load_workers} rep={rep} policy={policy} "
                        f"tok/s={result.get('decode_tok_s')} ms={result.get('generation_decode_ms')}"
                    )
            finally:
                stop_load(busy)

    summary = summarize(rows)
    (args.out_dir / "raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (args.out_dir / "summary.csv").open("w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "load_workers",
                "policy",
                "runs",
                "decode_tok_s_mean",
                "decode_tok_s_stdev",
                "generation_decode_ms_mean",
                "speedup_vs_same_load_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    write_summary_md(args.out_dir / "SUMMARY.md", args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
