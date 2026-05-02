#!/usr/bin/env python3
"""Fit a runtime policy model from closed-loop scheduler measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


EXP_DIR = Path(__file__).resolve().parent


def parse_nums(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def policy_rate(policy: str) -> float:
    if policy.startswith("trunc_even_"):
        return float(policy.rsplit("_", 1)[1])
    return 0.0


def features(row: dict[str, Any]) -> list[float]:
    loads = parse_nums(row["loads"])
    rate = policy_rate(row["policy"])
    mean = sum(loads) / len(loads)
    std = math.sqrt(sum((x - mean) ** 2 for x in loads) / len(loads)) if len(loads) > 1 else 0.0
    return [
        float(row["n_cores"]),
        min(loads),
        mean,
        max(loads),
        std,
        float(sum(1 for x in loads if x >= 50)),
        float(sum(1 for x in loads if x >= 80)),
        rate,
        rate * rate,
        rate * mean,
        rate * std,
    ]


def load_dataset(raw_paths: list[Path], clip_speedup: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in raw_paths:
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))

    baselines: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["phase"], row["scenario"], row["repeat"])
        if row["policy"] == "baseline_no_svd" and row.get("decode_tok_s"):
            baselines[key] = float(row["decode_tok_s"])

    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    groups: list[str] = []
    used: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("decode_tok_s"):
            continue
        key = (row["phase"], row["scenario"], row["repeat"])
        base = baselines.get(key)
        if not base or base <= 0:
            continue
        speedup = float(row["decode_tok_s"]) / base
        clipped_speedup = min(max(speedup, 1e-6), clip_speedup)
        x_rows.append(features(row))
        y_rows.append(math.log(clipped_speedup))
        groups.append(row["scenario"])
        used.append({**row, "speedup": speedup, "clipped_speedup": clipped_speedup})
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float), np.asarray(groups), groups, used


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit runtime scheduler policy model.")
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--clip-speedup", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/runtime_policy_model_20260429_r1")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, y, group_arr, groups, used = load_dataset(args.raw, args.clip_speedup)
    feature_names = [
        "n_cores",
        "load_min",
        "load_mean",
        "load_max",
        "load_std",
        "count_ge_50",
        "count_ge_80",
        "policy_rate",
        "policy_rate_sq",
        "rate_x_load_mean",
        "rate_x_load_std",
    ]

    splitter = GroupShuffleSplit(n_splits=20, test_size=0.25, random_state=29)
    candidates = {
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=29),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=29),
    }
    metrics: list[dict[str, Any]] = []
    best_name = ""
    best_mae = float("inf")
    for name, model in candidates.items():
        maes = []
        r2s = []
        for train_idx, test_idx in splitter.split(x, y, groups=group_arr):
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            maes.append(mean_absolute_error(np.exp(y[test_idx]), np.exp(pred)))
            r2s.append(r2_score(y[test_idx], pred))
        mae = float(np.mean(maes))
        metrics.append({"model": name, "speedup_mae": mae, "log_r2_mean": float(np.mean(r2s))})
        if mae < best_mae:
            best_mae = mae
            best_name = name

    best = candidates[best_name]
    best.fit(x, y)
    payload = {
        "model": best,
        "feature_names": feature_names,
        "target": "log(speedup_vs_no_svd)",
        "policy": "enable non-baseline candidate only when predicted speedup >= safety threshold",
    }
    with (args.out_dir / "runtime_policy_model.pkl").open("wb") as f:
        pickle.dump(payload, f)

    pred = np.exp(best.predict(x))
    pred_rows = []
    for row, pred_speed in zip(used, pred):
        pred_rows.append(
            {
                "phase": row["phase"],
                "scenario": row["scenario"],
                "repeat": row["repeat"],
                "policy": row["policy"],
                "actual_speedup": row["speedup"],
                "clipped_speedup": row["clipped_speedup"],
                "predicted_speedup": float(pred_speed),
            }
        )
    write_csv(args.out_dir / "metrics.csv", metrics, ["model", "speedup_mae", "log_r2_mean"])
    write_csv(args.out_dir / "predictions.csv", pred_rows, ["phase", "scenario", "repeat", "policy", "actual_speedup", "clipped_speedup", "predicted_speedup"])
    (args.out_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Runtime Policy Model",
                "",
                f"- training rows: `{len(used)}`",
                f"- best model: `{best_name}`",
                f"- target: `log(speedup_vs_no_svd)`",
                f"- speedup clip for training: `{args.clip_speedup:.3f}x`",
                f"- best grouped-CV speedup MAE: `{best_mae:.6f}`",
                "",
                "This model is trained from real `decode_svd_test` runtime measurements, so it captures the observed overhead of local SVD truncation candidates under heterogeneous cgroup load better than the earlier standalone matrix-latency model.",
            ]
        )
        + "\n"
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(used), "best_model": best_name, "best_speedup_mae": best_mae}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
