#!/usr/bin/env python3
"""Generate a Latin-hypercube based profile-measurement dataset.

The dataset describes the configurations to measure for a later microbenchmark:

    Y = f(q_cpu60, q_cpu61, ..., q_cpu67, Q)

where q_cpu* are per-core background load percentages and Q is a discrete
matrix-size / work-size index.  The script intentionally generates measurement
points only; it does not run the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EXP_DIR = Path(__file__).resolve().parent


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            values.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            values.append(int(part))
    return sorted(dict.fromkeys(values))


@dataclass(frozen=True)
class Sample:
    source: str
    loads: tuple[int, ...]
    q_pct: int

    def key(self) -> tuple[tuple[int, ...], int]:
        return self.loads, self.q_pct


def snap_to_grid(value: float, grid: list[int]) -> int:
    return min(grid, key=lambda item: (abs(item - value), item))


def unit_to_grid(value: float, grid: list[int]) -> int:
    idx = int(value * len(grid))
    return grid[min(idx, len(grid) - 1)]


def add_sample(samples: dict[tuple[tuple[int, ...], int], Sample], sample: Sample) -> None:
    samples.setdefault(sample.key(), sample)


def add_all_equal(samples: dict[tuple[tuple[int, ...], int], Sample], n_cpus: int, loads: list[int], q_values: list[int]) -> None:
    for q_pct in q_values:
        for load in loads:
            add_sample(samples, Sample("all_equal", tuple([load] * n_cpus), q_pct))


def add_single_core_sweeps(
    samples: dict[tuple[tuple[int, ...], int], Sample],
    n_cpus: int,
    loads: list[int],
    q_values: list[int],
) -> None:
    for q_pct in q_values:
        for cpu_idx in range(n_cpus):
            for load in loads:
                vector = [0] * n_cpus
                vector[cpu_idx] = load
                add_sample(samples, Sample(f"single_core_{cpu_idx}", tuple(vector), q_pct))


def add_group_patterns(
    samples: dict[tuple[tuple[int, ...], int], Sample],
    n_cpus: int,
    q_values: list[int],
    pattern_loads: list[int],
) -> None:
    split = max(1, n_cpus // 2)
    for q_pct in q_values:
        for major_load in pattern_loads:
            for minor_load in pattern_loads:
                vector = [major_load] * split + [minor_load] * (n_cpus - split)
                add_sample(samples, Sample("major_minor_grid", tuple(vector), q_pct))
                vector = [minor_load] * split + [major_load] * (n_cpus - split)
                add_sample(samples, Sample("minor_major_grid", tuple(vector), q_pct))


def add_shape_patterns(
    samples: dict[tuple[tuple[int, ...], int], Sample],
    n_cpus: int,
    loads: list[int],
    q_values: list[int],
) -> None:
    low = min(loads)
    high = max(loads)
    mid = snap_to_grid(50, loads)
    gradient = [snap_to_grid(i * 100 / max(1, n_cpus - 1), loads) for i in range(n_cpus)]
    inverse_gradient = list(reversed(gradient))
    alternating = [high if i % 2 else low for i in range(n_cpus)]
    inverse_alternating = [low if i % 2 else high for i in range(n_cpus)]

    named_vectors = [
        ("all_idle", [low] * n_cpus),
        ("all_mid", [mid] * n_cpus),
        ("all_busy", [high] * n_cpus),
        ("gradient", gradient),
        ("inverse_gradient", inverse_gradient),
        ("alternating", alternating),
        ("inverse_alternating", inverse_alternating),
    ]

    for q_pct in q_values:
        for name, vector in named_vectors:
            add_sample(samples, Sample(name, tuple(vector), q_pct))

        for busy_count in range(1, n_cpus + 1):
            vector = [high if i < busy_count else low for i in range(n_cpus)]
            add_sample(samples, Sample("prefix_busy", tuple(vector), q_pct))
            vector = [low if i < busy_count else high for i in range(n_cpus)]
            add_sample(samples, Sample("suffix_busy", tuple(vector), q_pct))


def lhs_samples(
    n_samples: int,
    n_dims: int,
    loads: list[int],
    q_values: list[int],
    seed: int,
) -> Iterable[Sample]:
    rng = random.Random(seed)
    # One independent shuffled Latin partition per dimension.
    dims: list[list[float]] = []
    for _ in range(n_dims):
        values = [(i + rng.random()) / n_samples for i in range(n_samples)]
        rng.shuffle(values)
        dims.append(values)

    for sample_idx in range(n_samples):
        raw = [dims[dim][sample_idx] for dim in range(n_dims)]
        loads_tuple = tuple(unit_to_grid(value, loads) for value in raw[:-1])
        q_pct = unit_to_grid(raw[-1], q_values)
        yield Sample("lhs", loads_tuple, q_pct)


def write_dataset(path: Path, cpus: list[int], rows: list[Sample]) -> None:
    fieldnames = ["sample_id", "source", "Q_pct"]
    fieldnames.extend(f"q_cpu{cpu}" for cpu in cpus)
    fieldnames.append("load_vector")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, sample in enumerate(rows):
            row = {
                "sample_id": idx,
                "source": sample.source,
                "Q_pct": sample.q_pct,
                "load_vector": ",".join(f"{cpu}:{load}" for cpu, load in zip(cpus, sample.loads)),
            }
            for cpu, load in zip(cpus, sample.loads):
                row[f"q_cpu{cpu}"] = load
            writer.writerow(row)


def write_report(path: Path, metadata: dict[str, object], rows: list[Sample]) -> None:
    by_source: dict[str, int] = {}
    by_q: dict[int, int] = {}
    for row in rows:
        by_source[row.source] = by_source.get(row.source, 0) + 1
        by_q[row.q_pct] = by_q.get(row.q_pct, 0) + 1

    lines = [
        "# LHS Profile Dataset",
        "",
        "This directory contains a measurement-plan dataset for fitting:",
        "",
        "```text",
        "Y = f(q_cpu60, q_cpu61, ..., q_cpu67, Q)",
        "```",
        "",
        "The dataset combines structured boundary points with Latin hypercube samples.",
        "It is a list of configurations to measure; it does not contain measured latency.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Counts",
        "",
        f"- total unique configurations: {len(rows)}",
        f"- requested LHS samples: {metadata['lhs_samples_requested']}",
        f"- duplicate configurations removed: {metadata['duplicates_removed']}",
        "",
        "## Count By Source",
        "",
        "| source | count |",
        "|---|---:|",
    ]
    for source, count in sorted(by_source.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {source} | {count} |")

    lines.extend(["", "## Count By Q", "", "| Q_pct | count |", "|---:|---:|"])
    for q_pct, count in sorted(by_q.items()):
        lines.append(f"| {q_pct} | {count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `all_equal`, `single_core_*`, `major_minor_grid`, and shape-pattern rows are deterministic boundary/structure points.",
            "- `lhs` rows provide high-dimensional coverage without enumerating the full grid.",
            "- The CSV should be consumed by a dedicated one-layer / one-matmul microbenchmark, not by the full-model decode benchmark.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an LHS profile-measurement dataset.")
    parser.add_argument("--cpus", default="60-67", help="CPU list/range, e.g. 60-67")
    parser.add_argument("--loads", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--q-values", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--lhs-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/lhs_profile_dataset_latest")
    args = parser.parse_args()

    cpus = parse_int_list(args.cpus)
    loads = parse_int_list(args.loads)
    q_values = parse_int_list(args.q_values)
    if not cpus:
        raise SystemExit("no CPUs specified")
    if not loads or min(loads) < 0 or max(loads) > 100:
        raise SystemExit("loads must be percentages in [0, 100]")
    if not q_values or min(q_values) < 0 or max(q_values) > 100:
        raise SystemExit("Q values must be percentages in [0, 100]")
    if args.lhs_samples <= 0:
        raise SystemExit("--lhs-samples must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[tuple[tuple[int, ...], int], Sample] = {}
    add_all_equal(samples, len(cpus), loads, q_values)
    add_single_core_sweeps(samples, len(cpus), loads, q_values)
    add_group_patterns(samples, len(cpus), q_values, [0, 20, 50, 80, 100])
    add_shape_patterns(samples, len(cpus), loads, q_values)

    before_lhs = len(samples)
    for sample in lhs_samples(args.lhs_samples, len(cpus) + 1, loads, q_values, args.seed):
        add_sample(samples, sample)

    rows = sorted(samples.values(), key=lambda item: (item.source != "lhs", item.source, item.q_pct, item.loads))
    dataset_path = args.out_dir / "profile_lhs_dataset.csv"
    metadata = {
        "cpus": cpus,
        "loads": loads,
        "q_values": q_values,
        "lhs_samples_requested": args.lhs_samples,
        "seed": args.seed,
        "structured_unique_before_lhs": before_lhs,
        "total_unique": len(rows),
        "duplicates_removed": before_lhs + args.lhs_samples - len(rows),
        "dataset_csv": str(dataset_path),
    }

    write_dataset(dataset_path, cpus, rows)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    write_report(args.out_dir / "LHS_PROFILE_DATASET_REPORT.md", metadata, rows)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
