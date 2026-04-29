#!/usr/bin/env python3
"""Fit a latency predictor from measured exp12 profile data.

The current measured data covers CPU-set/load-vector effects for one fixed
decode workload.  The feature table keeps Q_pct so the same model interface can
consume future LHS microbenchmark data, but existing measurements do not yet
support learning Q-dependence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Row:
    source: str
    name: str
    cpus: tuple[int, ...]
    loads: dict[int, int]
    q_pct: int
    latency_ms: float


def parse_cpus(spec: str) -> tuple[int, ...]:
    if not spec:
        return tuple()
    return tuple(sorted(int(item.strip()) for item in spec.split(",") if item.strip()))


def parse_load_vector(spec: str) -> dict[int, int]:
    loads: dict[int, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cpu_s, load_s = item.split(":", 1)
        loads[int(cpu_s)] = int(load_s)
    return loads


def read_single_core(path: Path, q_pct: int) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("status") != "ok":
                continue
            ms = float(raw["generation_decode_ms"])
            cpu = int(raw["cpu"])
            load = int(raw["load_pct"])
            rows.append(Row("single", f"single_cpu{cpu}_load{load}", (cpu,), {cpu: load}, q_pct, ms))
    return rows


def read_multi_core(path: Path, q_pct: int) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("status") != "ok":
                continue
            cpus = parse_cpus(raw["cpus"])
            load = int(raw["load_pct"])
            ms = float(raw["generation_decode_ms"])
            rows.append(Row("uniform_multi", f"multi_{'_'.join(map(str, cpus))}_load{load}", cpus, {cpu: load for cpu in cpus}, q_pct, ms))
    return rows


def read_heterogeneous(path: Path, q_pct: int) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            value = raw.get("measured_ms_median")
            if not value:
                continue
            cpus = parse_cpus(raw["cpus"])
            loads = parse_load_vector(raw["load_vector"])
            rows.append(Row("heterogeneous", raw["name"], cpus, loads, q_pct, float(value)))
    return rows


def load_rows(args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    if args.single_core_csv:
        rows.extend(read_single_core(args.single_core_csv, args.q_pct))
    if args.multi_core_csv:
        rows.extend(read_multi_core(args.multi_core_csv, args.q_pct))
    if args.heterogeneous_csv:
        rows.extend(read_heterogeneous(args.heterogeneous_csv, args.q_pct))
    return rows


def feature_names(cpus: list[int]) -> list[str]:
    names = ["Q_pct", "n_active"]
    names.extend(f"active_cpu{cpu}" for cpu in cpus)
    names.extend(f"q_cpu{cpu}" for cpu in cpus)
    names.extend(f"active_q_cpu{cpu}" for cpu in cpus)
    names.extend(
        [
            "load_min",
            "load_mean",
            "load_max",
            "load_std",
            "count_ge_50",
            "count_ge_80",
            "count_eq_100",
            "load_std_x_count_ge_80",
            "load_std_x_count_eq_100",
            "load_max_x_count_ge_80",
        ]
    )
    return names


def row_to_features(row: Row, cpus: list[int]) -> list[float]:
    active = {cpu: 1.0 if cpu in row.cpus else 0.0 for cpu in cpus}
    q_values = {cpu: float(row.loads.get(cpu, 0)) for cpu in cpus}
    active_loads = [q_values[cpu] for cpu in cpus if active[cpu] > 0.0]
    if active_loads:
        load_min = min(active_loads)
        load_mean = statistics.mean(active_loads)
        load_max = max(active_loads)
        load_std = statistics.pstdev(active_loads) if len(active_loads) > 1 else 0.0
        count_ge_50 = sum(1 for value in active_loads if value >= 50)
        count_ge_80 = sum(1 for value in active_loads if value >= 80)
        count_eq_100 = sum(1 for value in active_loads if value >= 100)
    else:
        load_min = load_mean = load_max = load_std = 0.0
        count_ge_50 = count_ge_80 = count_eq_100 = 0

    values: list[float] = [float(row.q_pct), float(len(row.cpus))]
    values.extend(active[cpu] for cpu in cpus)
    values.extend(q_values[cpu] for cpu in cpus)
    values.extend(active[cpu] * q_values[cpu] for cpu in cpus)
    values.extend(
        [
            load_min,
            load_mean,
            load_max,
            load_std,
            float(count_ge_50),
            float(count_ge_80),
            float(count_eq_100),
            load_std * float(count_ge_80),
            load_std * float(count_eq_100),
            load_max * float(count_ge_80),
        ]
    )
    return values


def build_matrix(rows: list[Row], cpus: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    X = np.asarray([row_to_features(row, cpus) for row in rows], dtype=float)
    y = np.asarray([row.latency_ms for row in rows], dtype=float)
    meta = [
        {
            "source": row.source,
            "name": row.name,
            "cpus": ",".join(map(str, row.cpus)),
            "load_vector": ",".join(f"{cpu}:{row.loads.get(cpu, 0)}" for cpu in row.cpus),
            "Q_pct": row.q_pct,
            "latency_ms": row.latency_ms,
        }
        for row in rows
    ]
    return X, y, meta


def models(random_state: int) -> dict[str, Any]:
    return {
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge_log": TransformedTargetRegressor(
            regressor=Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "extra_trees_log": TransformedTargetRegressor(
            regressor=ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rel = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-9)
    return {
        "mae_ms": float(mean_absolute_error(y_true, y_pred)),
        "median_ae_ms": float(median_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "median_ape": float(np.median(rel)),
        "p90_ape": float(np.quantile(rel, 0.9)),
        "max_ape": float(np.max(rel)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate an sklearn latency model.")
    default_root = EXP_DIR / "results/additivity_full_20260429_r1"
    default_hetero = EXP_DIR / "results/heterogeneous_additivity_20260429_r2/heterogeneous_additivity_error.csv"
    parser.add_argument("--single-core-csv", type=Path, default=default_root / "single_core.csv")
    parser.add_argument("--multi-core-csv", type=Path, default=default_root / "multi_core.csv")
    parser.add_argument("--heterogeneous-csv", type=Path, default=default_hetero)
    parser.add_argument("--cpus", default="60,61,62,63,64,65,66,67")
    parser.add_argument("--q-pct", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=20260429)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/latency_model_20260429_r1")
    args = parser.parse_args()

    cpus = parse_cpus(args.cpus)
    if not cpus:
        raise SystemExit("no CPUs specified")

    rows = load_rows(args)
    if len(rows) < 20:
        raise SystemExit(f"not enough measured rows: {len(rows)}")

    X, y, meta = build_matrix(rows, list(cpus))
    stratify = np.asarray(["slow" if row.latency_ms >= 500.0 else "normal" for row in rows], dtype=object)
    unique, counts = np.unique(stratify, return_counts=True)
    stratify_arg = stratify if np.all(counts >= 2) else None
    idx = np.arange(len(rows))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_arg,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = []
    for i, item in enumerate(meta):
        split = "test" if i in set(test_idx.tolist()) else "train"
        dataset_rows.append({"row_id": i, "split": split, **item})
    write_csv(
        args.out_dir / "supervised_dataset.csv",
        dataset_rows,
        ["row_id", "split", "source", "name", "cpus", "load_vector", "Q_pct", "latency_ms"],
    )

    result_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, model in models(args.random_state).items():
        model.fit(X[train_idx], y[train_idx])
        fitted[name] = model
        train_pred = model.predict(X[train_idx])
        test_pred = model.predict(X[test_idx])
        train_metrics = metrics(y[train_idx], train_pred)
        test_metrics = metrics(y[test_idx], test_pred)
        result_rows.append(
            {
                "model": name,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    best = min(result_rows, key=lambda row: (row["test_mape"], row["test_mae_ms"]))
    best_model_name = str(best["model"])
    best_model = fitted[best_model_name]
    pred = best_model.predict(X[test_idx])
    prediction_rows: list[dict[str, Any]] = []
    for row_id, pred_ms in zip(test_idx.tolist(), pred):
        actual = float(y[row_id])
        prediction_rows.append(
            {
                "row_id": row_id,
                **meta[row_id],
                "actual_ms": actual,
                "predicted_ms": float(pred_ms),
                "abs_error_ms": abs(float(pred_ms) - actual),
                "ape": abs(float(pred_ms) - actual) / max(actual, 1e-9),
            }
        )
    prediction_rows.sort(key=lambda row: float(row["ape"]), reverse=True)

    metric_fields = [
        "model",
        "train_rows",
        "test_rows",
        "train_mae_ms",
        "train_median_ae_ms",
        "train_mape",
        "train_median_ape",
        "train_p90_ape",
        "train_max_ape",
        "train_r2",
        "test_mae_ms",
        "test_median_ae_ms",
        "test_mape",
        "test_median_ape",
        "test_p90_ape",
        "test_max_ape",
        "test_r2",
    ]
    write_csv(args.out_dir / "model_metrics.csv", result_rows, metric_fields)
    write_csv(
        args.out_dir / "test_predictions.csv",
        prediction_rows,
        ["row_id", "source", "name", "cpus", "load_vector", "Q_pct", "latency_ms", "actual_ms", "predicted_ms", "abs_error_ms", "ape"],
    )

    with (args.out_dir / "best_model.pkl").open("wb") as f:
        pickle.dump({"model": best_model, "feature_names": feature_names(list(cpus)), "cpus": list(cpus)}, f)

    summary = {
        "rows": len(rows),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "feature_names": feature_names(list(cpus)),
        "best_model": best_model_name,
        "best_test_metrics": {key.removeprefix("test_"): value for key, value in best.items() if key.startswith("test_") and key != "test_rows"},
        "q_pct_note": "Current measured inputs use a fixed Q_pct; this run evaluates load/core prediction, not Q generalization.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Latency Model Report",
        "",
        "This report fits sklearn tabular regressors to the measured exp12 latency data.",
        "The current measured data covers CPU-set and load-vector variation for one fixed workload; `Q_pct` is present in the feature interface but fixed in this run.",
        "",
        "## Dataset",
        "",
        f"- rows: {len(rows)}",
        f"- train rows: {len(train_idx)}",
        f"- test rows: {len(test_idx)}",
        f"- CPUs: {','.join(map(str, cpus))}",
        "",
        "## Test Metrics",
        "",
        "| model | MAE ms | MAPE | median APE | p90 APE | max APE | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result_rows, key=lambda item: item["test_mape"]):
        lines.append(
            f"| {row['model']} | {row['test_mae_ms']:.4f} | {row['test_mape']:.4f} | "
            f"{row['test_median_ape']:.4f} | {row['test_p90_ape']:.4f} | {row['test_max_ape']:.4f} | {row['test_r2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Model",
            "",
            f"- model: `{best_model_name}`",
            f"- test MAE: {best['test_mae_ms']:.4f} ms",
            f"- test MAPE: {best['test_mape']:.4f}",
            f"- test median APE: {best['test_median_ape']:.4f}",
            f"- test p90 APE: {best['test_p90_ape']:.4f}",
            "",
            "## Worst Test Predictions",
            "",
            "| source | name | actual ms | predicted ms | APE |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in prediction_rows[:10]:
        lines.append(
            f"| {row['source']} | {row['name']} | {float(row['actual_ms']):.4f} | "
            f"{float(row['predicted_ms']):.4f} | {float(row['ape']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `supervised_dataset.csv`: normalized measured dataset and train/test split",
            "- `model_metrics.csv`: train/test metrics for all candidate models",
            "- `test_predictions.csv`: held-out predictions from the best model",
            "- `best_model.pkl`: pickled sklearn model plus feature metadata",
        ]
    )
    (args.out_dir / "LATENCY_MODEL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
