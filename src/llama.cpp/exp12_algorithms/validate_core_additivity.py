#!/usr/bin/env python3
"""Validate whether per-core effective throughput is approximately additive.

The validation model is:

    T(S, u) ~= 1 / sum_{i in S} (1 / T_i(u_i))

where T_i(u_i) is the time for core i, under background utilization u_i, to
run the same decode workload alone.  The script measures single-core curves
for utilization 0..100 step 10 by default, then measures selected multi-core
sets under the same utilization and reports prediction error.

This is deliberately a workload-level validation.  If the additivity error is
small enough, the later profile builder can use the same effective-throughput
model to synthesize Tmain/Ttail for core splits without enumerating every
utilization vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf"
DEFAULT_BINARY = ROOT / "build-release-current/decode_svd_test"
DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup/ce_ada_llama_6079")


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def cpu_spec(cpus: list[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def safe_name(prefix: str, cpus: list[int]) -> str:
    return f"{prefix}_{'_'.join(str(cpu) for cpu in sorted(cpus))}"


def run(cmd: list[str], *, env: dict[str, str] | None = None, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=timeout_s,
        errors="replace",
    )


def make_cgroup(cgroup_root: Path, name: str, cpus: list[int]) -> Path:
    cg = cgroup_root / name
    subprocess.run(["sudo", "-n", "mkdir", "-p", str(cg)], check=True)
    subprocess.run(["sudo", "-n", "sh", "-c", f"echo 0 > {shlex.quote(str(cg / 'cpuset.mems'))}"], check=True)
    subprocess.run(["sudo", "-n", "sh", "-c", f"echo {cpu_spec(cpus)} > {shlex.quote(str(cg / 'cpuset.cpus'))}"], check=True)
    return cg


def run_in_cgroup(cmd: list[str], cgroup: Path, env: dict[str, str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
    script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
    wrapped = [
        "sudo",
        "-n",
        "env",
        f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}",
        "bash",
        "-lc",
        script,
    ]
    return run(wrapped, env=env, timeout_s=timeout_s)


def start_load(cgroup_root: Path, cpus: list[int], load_pct: int, duration_s: int, out_log: Path) -> list[subprocess.Popen[str]]:
    if load_pct <= 0:
        return []
    procs: list[subprocess.Popen[str]] = []
    for cpu in cpus:
        cg = make_cgroup(cgroup_root, safe_name("add_load", [cpu]), [cpu])
        script = (
            f"echo $$ > {shlex.quote(str(cg / 'cgroup.procs'))}; "
            f"exec taskset -c {cpu} stress-ng --cpu 1 --cpu-load {load_pct} --timeout {duration_s}s"
        )
        log = out_log.open("ab")
        procs.append(
            subprocess.Popen(
                ["sudo", "-n", "bash", "-lc", script],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    time.sleep(1.0)
    return procs


def stop_load(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(["sudo", "-n", "pkill", "-x", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_decode_ms(output: str) -> float | None:
    match = re.search(r"generation_decode=([0-9.]+) ms", output)
    return float(match.group(1)) if match else None


def run_decode_once(
    binary: Path,
    model: Path,
    cpus: list[int],
    cgroup_root: Path,
    out_dir: Path,
    label: str,
    timeout_s: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / "build-release-current/bin")
    cg = make_cgroup(cgroup_root, safe_name("add_run", cpus), cpus)
    cmd = [
        "taskset",
        "-c",
        cpu_spec(cpus),
        str(binary),
        str(model),
        "1",
        str(len(cpus)),
        "0",
        "off",
        "off",
        "off",
        "off",
        "0.5",
        "0",
        "2",
        "off",
    ]
    started = time.time()
    try:
        completed = run_in_cgroup(cmd, cg, env, timeout_s=timeout_s)
        elapsed = time.time() - started
        output = completed.stdout
        status = "ok" if completed.returncode == 0 and parse_decode_ms(output) is not None else "bad_output"
        if completed.returncode != 0:
            status = f"returncode_{completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        status = "timeout"
    log_path = out_dir / f"{label}.log"
    log_path.write_text(output, errors="replace")
    return {
        "status": status,
        "elapsed_s": elapsed,
        "generation_decode_ms": parse_decode_ms(output),
        "log": str(log_path),
    }


def median(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return float(statistics.median(clean))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def default_combos(cpus: list[int]) -> list[list[int]]:
    combos: list[list[int]] = []
    if len(cpus) >= 2:
        combos.append(cpus[:2])
    if len(cpus) >= 4:
        combos.append(cpus[:4])
    if len(cpus) >= 6:
        combos.append(cpus[:6])
    if len(cpus) >= 8:
        combos.append(cpus[:8])
    if len(cpus) >= 4:
        combos.append([cpus[0], cpus[2]])
        combos.append([cpus[1], cpus[3]])
    return [sorted(set(item)) for item in combos]


def parse_combos(spec: str, cpus: list[int]) -> list[list[int]]:
    if spec == "default":
        return default_combos(cpus)
    return [parse_cpu_list(item) for item in spec.split(";") if item.strip()]


def summarize_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        float(row["relative_error"])
        for row in rows
        if row.get("relative_error") not in ("", None) and math.isfinite(float(row["relative_error"]))
    ]
    if not errors:
        return {"count": 0}
    return {
        "count": len(errors),
        "mean": statistics.mean(errors),
        "median": statistics.median(errors),
        "p90": statistics.quantiles(errors, n=10)[8] if len(errors) >= 10 else max(errors),
        "max": max(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate core throughput additivity.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--cpus", default="60-67")
    parser.add_argument("--loads", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--combos", default="default", help='semicolon separated CPU specs, or "default"')
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--load-duration-s", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/additivity_latest")
    args = parser.parse_args()

    cpus = parse_cpu_list(args.cpus)
    loads = [int(item) for item in args.loads.split(",") if item.strip()]
    combos = parse_combos(args.combos, cpus)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["sudo", "-n", "pkill", "-x", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    single_rows: list[dict[str, Any]] = []
    single_medians: dict[tuple[int, int], float] = {}
    for load in loads:
        for cpu in cpus:
            times: list[float] = []
            for rep in range(args.repeats):
                label = f"single_cpu{cpu}_load{load}_rep{rep}"
                load_procs = start_load(args.cgroup_root, [cpu], load, args.load_duration_s, args.out_dir / "stress.log")
                try:
                    result = run_decode_once(args.binary, args.model, [cpu], args.cgroup_root, args.out_dir, label, args.timeout_s)
                finally:
                    stop_load(load_procs)
                if result["generation_decode_ms"] is not None:
                    times.append(float(result["generation_decode_ms"]))
                single_rows.append({
                    "kind": "single",
                    "cpu": cpu,
                    "load_pct": load,
                    "repeat": rep,
                    "status": result["status"],
                    "generation_decode_ms": result["generation_decode_ms"],
                    "elapsed_s": result["elapsed_s"],
                    "log": result["log"],
                })
            med = median(times)
            if med is not None:
                single_medians[(cpu, load)] = med

    multi_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for load in loads:
        for combo in combos:
            measured_times: list[float] = []
            for rep in range(args.repeats):
                label = f"multi_{cpu_spec(combo).replace(',', '_')}_load{load}_rep{rep}"
                load_procs = start_load(args.cgroup_root, combo, load, args.load_duration_s, args.out_dir / "stress.log")
                try:
                    result = run_decode_once(args.binary, args.model, combo, args.cgroup_root, args.out_dir, label, args.timeout_s)
                finally:
                    stop_load(load_procs)
                if result["generation_decode_ms"] is not None:
                    measured_times.append(float(result["generation_decode_ms"]))
                multi_rows.append({
                    "kind": "multi",
                    "cpus": cpu_spec(combo),
                    "n_cpus": len(combo),
                    "load_pct": load,
                    "repeat": rep,
                    "status": result["status"],
                    "generation_decode_ms": result["generation_decode_ms"],
                    "elapsed_s": result["elapsed_s"],
                    "log": result["log"],
                })
            measured = median(measured_times)
            inv_sum = 0.0
            missing = []
            for cpu in combo:
                t = single_medians.get((cpu, load))
                if t is None or t <= 0:
                    missing.append(cpu)
                else:
                    inv_sum += 1.0 / t
            predicted = (1.0 / inv_sum) if inv_sum > 0 and not missing else None
            rel = abs(measured - predicted) / measured if measured and predicted else None
            error_rows.append({
                "cpus": cpu_spec(combo),
                "n_cpus": len(combo),
                "load_pct": load,
                "measured_ms": measured,
                "predicted_ms": predicted,
                "relative_error": rel,
                "missing_single_cpus": cpu_spec(missing),
            })

    write_csv(
        args.out_dir / "single_core.csv",
        single_rows,
        ["kind", "cpu", "load_pct", "repeat", "status", "generation_decode_ms", "elapsed_s", "log"],
    )
    write_csv(
        args.out_dir / "multi_core.csv",
        multi_rows,
        ["kind", "cpus", "n_cpus", "load_pct", "repeat", "status", "generation_decode_ms", "elapsed_s", "log"],
    )
    write_csv(
        args.out_dir / "additivity_error.csv",
        error_rows,
        ["cpus", "n_cpus", "load_pct", "measured_ms", "predicted_ms", "relative_error", "missing_single_cpus"],
    )

    summary = summarize_errors(error_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    verdict = "usable_with_calibration"
    if summary.get("count", 0) == 0:
        verdict = "invalid"
    elif summary["median"] <= 0.10 and summary["p90"] <= 0.20:
        verdict = "additive_ok"
    elif summary["median"] <= 0.20 and summary["p90"] <= 0.35:
        verdict = "usable_with_calibration"
    else:
        verdict = "direct_split_profile_required"

    report_lines = [
        "# Core Additivity Validation",
        "",
        f"- CPUs: `{cpu_spec(cpus)}`",
        f"- Loads: `{','.join(str(x) for x in loads)}`",
        f"- Repeats: `{args.repeats}`",
        f"- Combos: `{'; '.join(cpu_spec(c) for c in combos)}`",
        f"- Verdict: `{verdict}`",
        "",
        "## Error Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in ["count", "mean", "median", "p90", "max"]:
        value = summary.get(key)
        if isinstance(value, float):
            report_lines.append(f"| {key} | {value:.4f} |")
        else:
            report_lines.append(f"| {key} | {value} |")
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "- `additive_ok`: harmonic throughput sum is accurate enough for the scheduler profile.",
        "- `usable_with_calibration`: use the additive model with group-size/load calibration factors.",
        "- `direct_split_profile_required`: do not rely on additivity; profile common core splits directly.",
        "",
    ])
    (args.out_dir / "ADDITIVITY_REPORT.md").write_text("\n".join(report_lines))
    print(json.dumps({"out_dir": str(args.out_dir), "verdict": verdict, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

