#!/usr/bin/env python3
"""Validate the model scheduler under heterogeneous per-core load.

The experiment runs entirely on the isolated 60-79 CPU set.  For each scenario
it starts one stress-ng worker per loaded core with an individual --cpu-load,
then compares no-SVD decode against the local schedule selected from the
model-based profile.
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

from benchmark_model_schedule_cgroup import (
    DEFAULT_CGROUP_ROOT,
    ROOT,
    decode_once,
    fmt_float,
    make_cgroup,
    sudo_sh,
)
from build_model_profile import LatencyPredictor, build_profile, cpu_spec
from run_exp12_local import DEFAULT_BINARY, DEFAULT_MODEL, EXP_DIR, parse_cpu_list


def parse_load_vector(spec: str) -> list[int]:
    return [int(item) for item in spec.split(",") if item.strip()]


def scenario_defaults() -> list[dict[str, Any]]:
    return [
        {"scenario": "4c_ramp_light", "cpus": [60, 61, 62, 63], "loads": [0, 10, 20, 30]},
        {"scenario": "4c_mixed", "cpus": [60, 61, 62, 63], "loads": [0, 30, 60, 90]},
        {"scenario": "4c_front_hot", "cpus": [60, 61, 62, 63], "loads": [90, 70, 20, 0]},
        {"scenario": "4c_high", "cpus": [60, 61, 62, 63], "loads": [70, 80, 90, 100]},
        {"scenario": "6c_ramp_light", "cpus": [60, 61, 62, 63, 64, 65], "loads": [0, 10, 20, 30, 40, 50]},
        {"scenario": "6c_mixed", "cpus": [60, 61, 62, 63, 64, 65], "loads": [0, 20, 40, 60, 80, 100]},
        {"scenario": "6c_front_hot", "cpus": [60, 61, 62, 63, 64, 65], "loads": [100, 80, 60, 30, 10, 0]},
        {"scenario": "6c_high", "cpus": [60, 61, 62, 63, 64, 65], "loads": [50, 60, 70, 80, 90, 100]},
        {"scenario": "8c_ramp_light", "cpus": [60, 61, 62, 63, 64, 65, 66, 67], "loads": [0, 10, 20, 30, 40, 50, 60, 70]},
        {"scenario": "8c_mixed", "cpus": [60, 61, 62, 63, 64, 65, 66, 67], "loads": [0, 20, 40, 60, 80, 100, 30, 50]},
        {"scenario": "8c_front_hot", "cpus": [60, 61, 62, 63, 64, 65, 66, 67], "loads": [100, 90, 80, 70, 30, 20, 10, 0]},
        {"scenario": "8c_high", "cpus": [60, 61, 62, 63, 64, 65, 66, 67], "loads": [30, 40, 50, 60, 70, 80, 90, 100]},
    ]


def start_heterogeneous_load(cgroup: Path, cpus: list[int], loads: list[int]) -> list[subprocess.Popen[str]]:
    procs: list[subprocess.Popen[str]] = []
    for cpu, load in zip(cpus, loads):
        if load <= 0:
            continue
        cmd = [
            "taskset",
            "-c",
            str(cpu),
            "stress-ng",
            "--cpu",
            "1",
            "--cpu-load",
            str(load),
            "--cpu-method",
            "matrixprod",
            "--quiet",
        ]
        script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
        procs.append(
            subprocess.Popen(
                ["sudo", "-n", "bash", "-lc", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                text=True,
            )
        )
    if procs:
        time.sleep(1.0)
    return procs


def stop_heterogeneous_load(cgroup: Path, procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for proc in procs:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5.0)


def run_scheduler(profile: Path, out_dir: Path, scenario: str, quantum_ms: float) -> dict[str, Any]:
    data = json.loads(profile.read_text())
    out_json = out_dir / f"{scenario}.schedule.json"
    out_rates = out_dir / f"{scenario}.rates.txt"
    out_timeouts = out_dir / f"{scenario}.timeouts.txt"
    cmd = [
        "python3",
        str(EXP_DIR / "scheduler.py"),
        "--profile",
        str(profile),
        "--local-deadline-ms",
        str(data["local_deadline_ms"]),
        "--request-deadline-ms",
        str(data["request_deadline_ms"]),
        "--timeout-budget-ms",
        str(data.get("timeout_budget_ms", 0.0)),
        "--quantum-ms",
        str(quantum_ms),
        "--out-json",
        str(out_json),
        "--out-rates",
        str(out_rates),
        "--out-timeouts",
        str(out_timeouts),
    ]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT)
    if completed.returncode not in (0, 2):
        return {"status": f"scheduler_returncode_{completed.returncode}", "output": completed.stdout}
    try:
        obj = json.loads(out_json.read_text())
    except json.JSONDecodeError:
        return {"status": "bad_scheduler_json", "output": completed.stdout}
    obj["status"] = "ok" if completed.returncode == 0 else "infeasible"
    obj["schedule_json"] = str(out_json)
    obj["rates_file"] = str(out_rates)
    obj["timeouts_file"] = str(out_timeouts)
    return obj


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize(raw_rows: list[dict[str, Any]], schedule_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario = {row["scenario"]: row for row in schedule_rows}
    out: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in raw_rows}):
        base_vals = [
            float(row["decode_tok_s"])
            for row in raw_rows
            if row["scenario"] == scenario and row["policy"] == "baseline_no_svd" and row.get("decode_tok_s")
        ]
        sched_vals = [
            float(row["decode_tok_s"])
            for row in raw_rows
            if row["scenario"] == scenario and row["policy"] == "model_schedule" and row.get("decode_tok_s")
        ]
        base_ms = [
            float(row["generation_decode_ms"])
            for row in raw_rows
            if row["scenario"] == scenario and row["policy"] == "baseline_no_svd" and row.get("generation_decode_ms")
        ]
        sched_ms = [
            float(row["generation_decode_ms"])
            for row in raw_rows
            if row["scenario"] == scenario and row["policy"] == "model_schedule" and row.get("generation_decode_ms")
        ]
        schedule = by_scenario.get(scenario, {})
        base = mean(base_vals)
        sched = mean(sched_vals)
        base_med = median(base_vals)
        sched_med = median(sched_vals)
        out.append(
            {
                "scenario": scenario,
                "n_cores": schedule.get("n_cores"),
                "cpus": schedule.get("cpus"),
                "loads": schedule.get("loads"),
                "baseline_runs": len(base_vals),
                "schedule_runs": len(sched_vals),
                "baseline_tok_s": base,
                "schedule_tok_s": sched,
                "speedup_vs_baseline": sched / base if sched is not None and base else None,
                "baseline_tok_s_median": base_med,
                "schedule_tok_s_median": sched_med,
                "speedup_vs_baseline_median": sched_med / base_med if sched_med is not None and base_med else None,
                "baseline_decode_ms": mean(base_ms),
                "schedule_decode_ms": mean(sched_ms),
                "baseline_decode_ms_median": median(base_ms),
                "schedule_decode_ms_median": median(sched_ms),
                "mode": schedule.get("mode"),
                "p": schedule.get("p"),
                "major_cpus": schedule.get("major_cpus"),
                "minor_cpus": schedule.get("minor_cpus"),
                "clipped_layers": schedule.get("clipped_layers"),
                "max_rate": schedule.get("max_rate"),
                "estimated_total_ms": schedule.get("estimated_total_ms"),
                "estimated_speedup": schedule.get("estimated_speedup"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark heterogeneous-load scheduler effectiveness in cgroups.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--latency-model", type=Path, default=EXP_DIR / "results/latency_model_20260429_r1/best_model.pkl")
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--scenarios-json", type=Path)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--rates", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--timeout-budget-ms", type=float, default=8.0)
    parser.add_argument("--request-deadline-factor", type=float, default=1.04)
    parser.add_argument("--local-deadline-factor", type=float, default=1.01)
    parser.add_argument("--quantum-ms", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/heterogeneous_schedule_effectiveness_20260429_r1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictor = LatencyPredictor(args.latency_model)
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    scenarios = json.loads(args.scenarios_json.read_text()) if args.scenarios_json else scenario_defaults()

    sudo_sh("pkill -x stress-ng || true", timeout_s=5.0)
    raw_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []

    try:
        for scenario_obj in scenarios:
            scenario = str(scenario_obj["scenario"])
            cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
            loads = [int(load) for load in scenario_obj["loads"]]
            if len(cpus) != len(loads):
                raise ValueError(f"{scenario}: cpus/load length mismatch")
            util = {cpu: load for cpu, load in zip(cpus, loads)}
            idle_reference = predictor.predict_ms(cpus, {cpu: 0 for cpu in cpus}, 100)
            profile = build_profile(
                predictor=predictor,
                cpus=cpus,
                util=util,
                n_layers=28,
                rates=rates,
                q_pct=100,
                timeout_budget_ms=args.timeout_budget_ms,
                request_deadline_factor=args.request_deadline_factor,
                local_deadline_factor=args.local_deadline_factor,
                reference_full_local_ms=idle_reference,
                tx_base_ms=2.0,
                tx_per_layer_ms=0.15,
                end_per_layer_ms=1.0,
            )
            profile_path = args.out_dir / f"{scenario}.profile.json"
            profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
            schedule = run_scheduler(profile_path, args.out_dir, scenario, args.quantum_ms)
            local = schedule.get("local") or {}
            major = [int(cpu) for cpu in (local.get("major_cpus") or [])]
            minor = [int(cpu) for cpu in (local.get("minor_cpus") or [])]
            sched_rates = [float(rate) for rate in (schedule.get("rates") or [])]
            clipped = sum(1 for rate in sched_rates if rate > 0.0)
            full_local_ms = float(profile["full_local_ms"])
            total_ms = schedule.get("total_ms")
            schedule_rows.append(
                {
                    "scenario": scenario,
                    "n_cores": len(cpus),
                    "cpus": cpu_spec(cpus),
                    "loads": ",".join(str(load) for load in loads),
                    "mode": schedule.get("mode"),
                    "feasible": schedule.get("feasible"),
                    "p": local.get("p"),
                    "major_cpus": cpu_spec(major),
                    "minor_cpus": cpu_spec(minor),
                    "clipped_layers": clipped,
                    "max_rate": max(sched_rates, default=0.0),
                    "estimated_total_ms": total_ms,
                    "estimated_full_local_ms": full_local_ms,
                    "estimated_speedup": full_local_ms / float(total_ms) if total_ms else None,
                    "rates_file": schedule.get("rates_file"),
                    "timeouts_file": schedule.get("timeouts_file"),
                }
            )

            run_cgroup = make_cgroup(args.cgroup_root, f"exp12_hetero_run_{scenario}", cpus)
            load_cgroup = make_cgroup(args.cgroup_root, f"exp12_hetero_load_{scenario}", cpus)
            for rep in range(args.repeats):
                load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
                try:
                    baseline = decode_once(args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s)
                    baseline.update(
                        {
                            "scenario": scenario,
                            "repeat": rep,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "policy": "baseline_no_svd",
                            "schedule_mode": "",
                            "clipped_layers": 0,
                        }
                    )
                    raw_rows.append(baseline)

                    if schedule.get("mode") == "local" and clipped == 0:
                        scheduled = dict(baseline)
                        scheduled.update(
                            {
                                "policy": "model_schedule",
                                "status": "same_as_baseline_no_svd",
                                "schedule_mode": schedule.get("mode"),
                                "clipped_layers": clipped,
                            }
                        )
                    elif schedule.get("mode") == "local":
                        scheduled = decode_once(
                            args.binary,
                            args.model,
                            run_cgroup,
                            cpus,
                            args.tokens,
                            args.timeout_s,
                            rates=Path(str(schedule["rates_file"])),
                            group_a=cpu_spec(major) if major else "off",
                            group_b=cpu_spec(minor) if minor else "off",
                            group_a_share=0.75,
                            layer_timeouts=Path(str(schedule["timeouts_file"])),
                        )
                        scheduled.update(
                            {
                                "scenario": scenario,
                                "repeat": rep,
                                "n_cores": len(cpus),
                                "cpus": cpu_spec(cpus),
                                "loads": ",".join(str(load) for load in loads),
                                "policy": "model_schedule",
                                "schedule_mode": schedule.get("mode"),
                                "clipped_layers": clipped,
                            }
                        )
                    else:
                        scheduled = {
                            "scenario": scenario,
                            "repeat": rep,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "policy": "model_schedule",
                            "status": f"skipped_{schedule.get('mode')}_no_adb",
                            "schedule_mode": schedule.get("mode"),
                            "clipped_layers": clipped,
                        }
                    raw_rows.append(scheduled)
                finally:
                    stop_heterogeneous_load(load_cgroup, load_procs)
    finally:
        sudo_sh("pkill -x stress-ng || true", timeout_s=5.0)

    summary_rows = summarize(raw_rows, schedule_rows)
    write_csv(
        args.out_dir / "raw.csv",
        raw_rows,
        [
            "scenario",
            "repeat",
            "n_cores",
            "cpus",
            "loads",
            "policy",
            "status",
            "schedule_mode",
            "clipped_layers",
            "decode_tok_s",
            "generation_decode_ms",
            "prefill_decode_ms",
            "elapsed_s",
            "succeed",
            "top1_id",
            "top1_piece",
            "output_tail",
        ],
    )
    write_csv(
        args.out_dir / "schedule_summary.csv",
        schedule_rows,
        [
            "scenario",
            "n_cores",
            "cpus",
            "loads",
            "mode",
            "feasible",
            "p",
            "major_cpus",
            "minor_cpus",
            "clipped_layers",
            "max_rate",
            "estimated_total_ms",
            "estimated_full_local_ms",
            "estimated_speedup",
            "rates_file",
            "timeouts_file",
        ],
    )
    write_csv(
        args.out_dir / "summary.csv",
        summary_rows,
        [
            "scenario",
            "n_cores",
            "cpus",
            "loads",
            "baseline_runs",
            "schedule_runs",
            "baseline_tok_s",
            "schedule_tok_s",
            "speedup_vs_baseline",
            "baseline_tok_s_median",
            "schedule_tok_s_median",
            "speedup_vs_baseline_median",
            "baseline_decode_ms",
            "schedule_decode_ms",
            "baseline_decode_ms_median",
            "schedule_decode_ms_median",
            "mode",
            "p",
            "major_cpus",
            "minor_cpus",
            "clipped_layers",
            "max_rate",
            "estimated_total_ms",
            "estimated_speedup",
        ],
    )

    lines = [
        "# Heterogeneous Load Scheduler Effectiveness",
        "",
        f"- cgroup root: `{args.cgroup_root}`",
        f"- repeats: `{args.repeats}`",
        f"- tokens per decode run: `{args.tokens}`",
        "- baseline: no SVD, same core set and same heterogeneous background load",
        "- model_schedule: model profile + DP scheduler; edge/end offload is skipped because this experiment does not use adb",
        "",
        "## Result",
        "",
        "| scenario | cores | loads | mode | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['scenario']}` | {row['n_cores']} | `{row['loads']}` | `{row['mode']}` | "
            f"`{row['major_cpus']}` | `{row['minor_cpus']}` | {row['clipped_layers']} | "
            f"{fmt_float(row['baseline_tok_s_median'])} | {fmt_float(row['schedule_tok_s_median'])} | "
            f"{fmt_float(row['speedup_vs_baseline_median'], digits=3, suffix='x')} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: aggregated throughput and speedup.",
            "- `schedule_summary.csv`: scheduler choices and estimated latencies.",
            "- `raw.csv`: every decode run with status and output tail.",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"out_dir": str(args.out_dir), "scenarios": len(scenarios), "raw_rows": len(raw_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
