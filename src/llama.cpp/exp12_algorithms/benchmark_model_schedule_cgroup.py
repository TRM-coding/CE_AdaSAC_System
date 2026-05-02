#!/usr/bin/env python3
"""Benchmark model-scheduler outputs inside the isolated 60-79 cgroup.

This script validates local schedules generated from model-based profiles.
Schedules that require edge/end offload are recorded but skipped because this
experiment intentionally does not use adb or a remote server.
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

from run_exp12_local import DEFAULT_BINARY, DEFAULT_MODEL, EXP_DIR, cpu_spec, parse_cpu_list, parse_decode_output


DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup/ce_ada_llama_6079")
ROOT = Path(__file__).resolve().parents[3]


def sudo_sh(script: str, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sudo", "-n", "bash", "-lc", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        errors="replace",
    )


def read_cgroup_file(path: Path, default: str) -> str:
    completed = sudo_sh(f"cat {shlex.quote(str(path))}", timeout_s=5.0)
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else default


def make_cgroup(root: Path, name: str, cpus: list[int]) -> Path:
    cg = root / name
    mems = read_cgroup_file(root / "cpuset.mems.effective", "0")
    sudo_sh(
        f"mkdir -p {shlex.quote(str(cg))}; "
        f"echo {shlex.quote(mems)} > {shlex.quote(str(cg / 'cpuset.mems'))}; "
        f"echo {shlex.quote(cpu_spec(cpus))} > {shlex.quote(str(cg / 'cpuset.cpus'))}"
    )
    return cg


def run_in_cgroup(cmd: list[str], cgroup: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / "build-release-current/bin")
    exports = f"export LD_LIBRARY_PATH={shlex.quote(env['LD_LIBRARY_PATH'])}; "
    script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; cd {shlex.quote(str(ROOT))}; {exports} exec {shlex.join(cmd)}"
    return sudo_sh(script, timeout_s=timeout_s)


def start_load(cgroup: Path, cpus: list[int], load_pct: int) -> subprocess.Popen[str] | None:
    if load_pct <= 0:
        return None
    cmd = [
        "taskset",
        "-c",
        cpu_spec(cpus),
        "stress-ng",
        "--cpu",
        str(len(cpus)),
        "--cpu-load",
        str(load_pct),
        "--cpu-method",
        "matrixprod",
        "--quiet",
    ]
    script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
    proc = subprocess.Popen(["sudo", "-n", "bash", "-lc", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1.0)
    return proc


def stop_load(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    subprocess.run(["sudo", "-n", "pkill", "-x", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    if completed.returncode != 0:
        return {"status": f"scheduler_returncode_{completed.returncode}", "output": completed.stdout}
    obj = json.loads(out_json.read_text())
    obj["schedule_json"] = str(out_json)
    obj["rates_file"] = str(out_rates)
    obj["timeouts_file"] = str(out_timeouts)
    return obj


def decode_once(
    binary: Path,
    model: Path,
    run_cgroup: Path,
    run_cpus: list[int],
    tokens: int,
    timeout_s: float,
    rates: Path | str = "off",
    group_a: str = "off",
    group_b: str = "off",
    group_a_share: float = 0.75,
    layer_timeouts: Path | str = "off",
    tail_mode: str = "off",
) -> dict[str, Any]:
    cmd = [
        "taskset",
        "-c",
        cpu_spec(run_cpus),
        str(binary),
        str(model),
        str(tokens),
        str(len(run_cpus)),
        "0",
        "off",
        str(rates),
        group_a,
        group_b,
        str(group_a_share),
        "0",
        "2",
        str(layer_timeouts),
        tail_mode,
    ]
    started = time.time()
    try:
        completed = run_in_cgroup(cmd, run_cgroup, timeout_s=timeout_s)
        output = completed.stdout
        parsed = parse_decode_output(output)
        if parsed.get("decode_tok_s") and parsed.get("succeed"):
            status = "ok" if completed.returncode == 0 else f"ok_returncode_{completed.returncode}"
        else:
            status = f"returncode_{completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        parsed = parse_decode_output(output)
        status = "timeout"
    parsed.update({"status": status, "elapsed_s": time.time() - started, "output_tail": "\n".join(output.splitlines()[-24:])})
    return parsed


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(int(row["load_pct"]), row["policy"]) for row in rows})
    baseline: dict[int, float] = {}
    for load, policy in keys:
        vals = [float(row["decode_tok_s"]) for row in rows if int(row["load_pct"]) == load and row["policy"] == policy and row.get("decode_tok_s")]
        if policy == "baseline_no_svd" and vals:
            baseline[load] = statistics.fmean(vals)
    for load, policy in keys:
        items = [row for row in rows if int(row["load_pct"]) == load and row["policy"] == policy]
        vals = [float(row["decode_tok_s"]) for row in items if row.get("decode_tok_s")]
        ms_vals = [float(row["generation_decode_ms"]) for row in items if row.get("generation_decode_ms")]
        avg = mean(vals)
        base = baseline.get(load)
        out.append(
            {
                "load_pct": load,
                "policy": policy,
                "runs": len(vals),
                "tok_s_mean": avg,
                "decode_ms_mean": mean(ms_vals),
                "speedup_vs_baseline_no_svd": avg / base if avg is not None and base else None,
            }
        )
    return out


def fmt_float(value: Any, digits: int = 6, suffix: str = "") -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}g}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model schedule effectiveness in isolated cgroups.")
    parser.add_argument("--profiles-dir", type=Path, default=EXP_DIR / "results/model_profiles_rerun_cgroup_20260429")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--run-cpus", default="60-67")
    parser.add_argument("--load-cpus", default="60-67")
    parser.add_argument("--loads", default="0,20,40,50,80,100")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--quantum-ms", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/model_schedule_effectiveness_cgroup_20260429")
    args = parser.parse_args()

    run_cpus = parse_cpu_list(args.run_cpus)
    load_cpus = parse_cpu_list(args.load_cpus)
    loads = [int(item) for item in args.loads.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_cgroup = make_cgroup(args.cgroup_root, "exp12_model_effect_run", run_cpus)
    load_cgroup = make_cgroup(args.cgroup_root, "exp12_model_effect_load", load_cpus)

    subprocess.run(["sudo", "-n", "pkill", "-x", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    try:
        for load in loads:
            scenario = f"load_{load}"
            profile = args.profiles_dir / f"profile_{scenario}.json"
            schedule = run_scheduler(profile, args.out_dir, scenario, args.quantum_ms)
            local = schedule.get("local") or {}
            mode = schedule.get("mode")
            major = local.get("major_cpus") or []
            minor = local.get("minor_cpus") or []
            rates = schedule.get("rates") or []
            clipped = sum(1 for rate in rates if float(rate) > 0.0)
            schedule_rows.append(
                {
                    "load_pct": load,
                    "mode": mode,
                    "p": local.get("p"),
                    "major_cpus": cpu_spec(major),
                    "minor_cpus": cpu_spec(minor),
                    "clipped_layers": clipped,
                    "max_rate": max([float(rate) for rate in rates], default=0.0),
                    "estimated_total_ms": schedule.get("total_ms"),
                    "rates_file": schedule.get("rates_file"),
                    "timeouts_file": schedule.get("timeouts_file"),
                }
            )

            for rep in range(args.repeats):
                proc = start_load(load_cgroup, load_cpus, load)
                try:
                    baseline = decode_once(args.binary, args.model, run_cgroup, run_cpus, args.tokens, args.timeout_s)
                    baseline.update({"load_pct": load, "repeat": rep, "policy": "baseline_no_svd", "schedule_mode": "", "clipped_layers": 0})
                    rows.append(baseline)

                    if mode == "local" and clipped == 0:
                        scheduled = dict(baseline)
                        scheduled.update(
                            {
                                "load_pct": load,
                                "repeat": rep,
                                "policy": "model_schedule",
                                "status": "same_as_baseline_no_svd",
                                "schedule_mode": mode,
                                "clipped_layers": clipped,
                            }
                        )
                        rows.append(scheduled)
                    elif mode == "local":
                        if minor:
                            group_a = cpu_spec(major)
                            group_b = cpu_spec(minor)
                        else:
                            group_a = "off"
                            group_b = "off"
                        scheduled = decode_once(
                            args.binary,
                            args.model,
                            run_cgroup,
                            run_cpus,
                            args.tokens,
                            args.timeout_s,
                            rates=Path(str(schedule["rates_file"])),
                            group_a=group_a,
                            group_b=group_b,
                            group_a_share=0.75,
                            layer_timeouts=Path(str(schedule["timeouts_file"])),
                        )
                        scheduled.update({"load_pct": load, "repeat": rep, "policy": "model_schedule", "schedule_mode": mode, "clipped_layers": clipped})
                        rows.append(scheduled)
                    else:
                        rows.append({"load_pct": load, "repeat": rep, "policy": "model_schedule", "status": "skipped_edge_end_no_adb", "schedule_mode": mode, "clipped_layers": clipped})
                finally:
                    stop_load(proc)
    finally:
        subprocess.run(["sudo", "-n", "pkill", "-x", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    summary = summarize(rows)
    write_csv(
        args.out_dir / "raw.csv",
        rows,
        ["load_pct", "repeat", "policy", "status", "schedule_mode", "clipped_layers", "decode_tok_s", "generation_decode_ms", "prefill_decode_ms", "elapsed_s", "succeed", "top1_id", "top1_piece", "output_tail"],
    )
    write_csv(
        args.out_dir / "schedule_summary.csv",
        schedule_rows,
        ["load_pct", "mode", "p", "major_cpus", "minor_cpus", "clipped_layers", "max_rate", "estimated_total_ms", "rates_file", "timeouts_file"],
    )
    write_csv(
        args.out_dir / "summary.csv",
        summary,
        ["load_pct", "policy", "runs", "tok_s_mean", "decode_ms_mean", "speedup_vs_baseline_no_svd"],
    )

    lines = [
        "# Model Schedule Effectiveness Rerun",
        "",
        "All decode and load processes were launched through sudo into child cgroups under `/sys/fs/cgroup/ce_ada_llama_6079`.",
        "",
        f"- run CPUs: `{cpu_spec(run_cpus)}`",
        f"- load CPUs: `{cpu_spec(load_cpus)}`",
        f"- loads: `{args.loads}`",
        f"- repeats: `{args.repeats}`",
        f"- tokens: `{args.tokens}`",
        "",
        "## Summary",
        "",
        "| load | policy | runs | tok/s mean | decode ms mean | speedup vs no-SVD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['load_pct']} | `{row['policy']}` | {row['runs']} | "
            f"{fmt_float(row['tok_s_mean'])} | "
            f"{fmt_float(row['decode_ms_mean'])} | "
            f"{fmt_float(row['speedup_vs_baseline_no_svd'], digits=3, suffix='x')} |"
        )
    lines.append("")
    lines.append("Schedules with `edge_end` are skipped in actual decode because this rerun intentionally does not use adb or a remote server.")
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows), "summary_rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
