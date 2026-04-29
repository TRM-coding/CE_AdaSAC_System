#!/usr/bin/env python3
"""Validate additive throughput under heterogeneous per-core load vectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from validate_core_additivity import (
    DEFAULT_BINARY,
    DEFAULT_CGROUP_ROOT,
    DEFAULT_MODEL,
    EXP_DIR,
    cpu_spec,
    parse_cpu_list,
    run_decode_once,
    start_load,
    stop_load,
    write_csv,
)


def load_single_core_times(single_core_csv: Path) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    with single_core_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            ms_s = row.get("generation_decode_ms")
            if not ms_s:
                continue
            ms = float(ms_s)
            if ms <= 0.0:
                continue
            grouped.setdefault((int(row["cpu"]), int(row["load_pct"])), []).append(ms)
    return {key: float(statistics.median(values)) for key, values in grouped.items() if values}


def nearest_single_time(single_times: dict[tuple[int, int], float], cpu: int, load: int) -> float:
    exact = single_times.get((cpu, load))
    if exact is not None:
        return exact
    candidates = sorted((abs(load - u), u, ms) for (c, u), ms in single_times.items() if c == cpu)
    if not candidates:
        raise KeyError(f"no single-core measurements for CPU {cpu}")
    return candidates[0][2]


def parse_vector(spec: str) -> tuple[list[int], dict[int, int]]:
    util: dict[int, int] = {}
    cpus: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cpu_s, load_s = item.split(":", 1)
        cpu = int(cpu_s)
        load = int(load_s)
        cpus.append(cpu)
        util[cpu] = load
    return sorted(cpus), util


def default_vectors() -> list[tuple[str, list[int], dict[int, int]]]:
    raw = [
        ("2core_0_50", "60:0,61:50"),
        ("2core_20_80", "60:20,61:80"),
        ("2core_0_100", "60:0,61:100"),
        ("4core_0_0_50_50", "60:0,61:0,62:50,63:50"),
        ("4core_0_20_60_100", "60:0,61:20,62:60,63:100"),
        ("4core_10_30_70_90", "60:10,61:30,62:70,63:90"),
        ("6core_mixed", "60:0,61:0,62:20,63:50,64:80,65:100"),
        ("8core_gradient", "60:0,61:10,62:20,63:30,64:60,65:70,66:90,67:100"),
        ("8core_major_idle_minor_busy", "60:0,61:0,62:0,63:0,64:80,65:80,66:100,67:100"),
        ("8core_alternating", "60:0,61:100,62:10,63:90,64:20,65:80,66:30,67:70"),
    ]
    vectors = []
    for name, spec in raw:
        cpus, util = parse_vector(spec)
        vectors.append((name, cpus, util))
    return vectors


def load_vectors_from_file(path: Path) -> list[tuple[str, list[int], dict[int, int]]]:
    vectors = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("name") or f"vector_{len(vectors)}"
            cpus, util = parse_vector(row["vector"])
            vectors.append((name, cpus, util))
    return vectors


def predict_ms(single_times: dict[tuple[int, int], float], cpus: list[int], util: dict[int, int]) -> tuple[float | None, str]:
    inv_sum = 0.0
    missing = []
    for cpu in cpus:
        try:
            t = nearest_single_time(single_times, cpu, util[cpu])
        except KeyError:
            missing.append(cpu)
            continue
        inv_sum += 1.0 / t
    if missing or inv_sum <= 0.0:
        return None, cpu_spec(missing)
    return 1.0 / inv_sum, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate heterogeneous-load additivity.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--single-core-csv", required=True, type=Path)
    parser.add_argument("--vectors-csv", type=Path, help='Optional CSV with columns: name,vector. Vector format: "60:0,61:50"')
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--load-duration-s", type=int, default=600)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/heterogeneous_additivity_latest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["sudo", "-n", "pkill", "-x", "stress-ng"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    single_times = load_single_core_times(args.single_core_csv)
    vectors = load_vectors_from_file(args.vectors_csv) if args.vectors_csv else default_vectors()

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for name, cpus, util in vectors:
        measured: list[float] = []
        for rep in range(args.repeats):
            load_procs = []
            try:
                for cpu in cpus:
                    load_procs.extend(
                        start_load(
                            args.cgroup_root,
                            [cpu],
                            util[cpu],
                            args.load_duration_s,
                            args.out_dir / "stress.log",
                        )
                    )
                label = f"{name}_rep{rep}"
                result = run_decode_once(args.binary, args.model, cpus, args.cgroup_root, args.out_dir, label, args.timeout_s)
            finally:
                stop_load(load_procs)
            if result["generation_decode_ms"] is not None:
                measured.append(float(result["generation_decode_ms"]))
            raw_rows.append(
                {
                    "name": name,
                    "cpus": cpu_spec(cpus),
                    "load_vector": ",".join(f"{cpu}:{util[cpu]}" for cpu in cpus),
                    "repeat": rep,
                    "status": result["status"],
                    "generation_decode_ms": result["generation_decode_ms"],
                    "elapsed_s": result["elapsed_s"],
                    "log": result["log"],
                }
            )
        measured_med = float(statistics.median(measured)) if measured else None
        predicted, missing = predict_ms(single_times, cpus, util)
        rel = abs(measured_med - predicted) / measured_med if measured_med and predicted else None
        summary_rows.append(
            {
                "name": name,
                "cpus": cpu_spec(cpus),
                "n_cpus": len(cpus),
                "load_vector": ",".join(f"{cpu}:{util[cpu]}" for cpu in cpus),
                "measured_ms_median": measured_med,
                "predicted_ms": predicted,
                "relative_error": rel,
                "missing_single_cpus": missing,
                "repeats": args.repeats,
            }
        )

    write_csv(
        args.out_dir / "heterogeneous_raw.csv",
        raw_rows,
        ["name", "cpus", "load_vector", "repeat", "status", "generation_decode_ms", "elapsed_s", "log"],
    )
    write_csv(
        args.out_dir / "heterogeneous_additivity_error.csv",
        summary_rows,
        [
            "name",
            "cpus",
            "n_cpus",
            "load_vector",
            "measured_ms_median",
            "predicted_ms",
            "relative_error",
            "missing_single_cpus",
            "repeats",
        ],
    )

    errors = [
        float(row["relative_error"])
        for row in summary_rows
        if row["relative_error"] is not None and math.isfinite(float(row["relative_error"]))
    ]
    summary = {
        "count": len(errors),
        "mean": statistics.mean(errors) if errors else None,
        "median": statistics.median(errors) if errors else None,
        "max": max(errors) if errors else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Heterogeneous Core Additivity Validation",
        "",
        f"- Single-core source: `{args.single_core_csv}`",
        f"- Repeats: `{args.repeats}`",
        "",
        "| name | cpus | load vector | measured ms | predicted ms | relative error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        rel_s = "" if row["relative_error"] is None else f"{float(row['relative_error']):.4f}"
        meas_s = "" if row["measured_ms_median"] is None else f"{float(row['measured_ms_median']):.4f}"
        pred_s = "" if row["predicted_ms"] is None else f"{float(row['predicted_ms']):.4f}"
        lines.append(
            f"| `{row['name']}` | `{row['cpus']}` | `{row['load_vector']}` | {meas_s} | {pred_s} | {rel_s} |"
        )
    lines.extend(["", "## Summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    (args.out_dir / "HETEROGENEOUS_ADDITIVITY_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"out_dir": str(args.out_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())

