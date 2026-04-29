#!/usr/bin/env python3
"""Load-aware SVD scheduling algorithms from algorithm.pdf chapter 5.

The module intentionally keeps the scheduler independent from llama.cpp.  It
consumes a small JSON layer profile, solves the local DP problem, optionally
searches a suffix offload split point, and writes the per-layer SVD rate file
that `decode_svd_test` already accepts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


INF = float("inf")


@dataclass(frozen=True)
class Candidate:
    rate: float
    main_ms: float
    loss: float
    weight: float
    tail_ms: float = 0.0

    @property
    def clipped(self) -> int:
        return 1 if self.rate > 0.0 else 0


@dataclass(frozen=True)
class LayerProfile:
    layer: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class CoreSplitProfile:
    split_id: str
    p: int
    major_cpus: tuple[int, ...]
    minor_cpus: tuple[int, ...]
    layers: tuple[LayerProfile, ...]


@dataclass
class LayerDecision:
    layer: int
    rate: float
    main_ms: float
    loss: float
    weight: float
    tail_ms: float = 0.0
    timeout_ms: float = 0.0


@dataclass
class LocalScheduleResult:
    feasible: bool
    deadline_ms: float
    split_id: str | None = None
    p: int | None = None
    major_cpus: list[int] | None = None
    minor_cpus: list[int] | None = None
    total_main_ms: float = INF
    total_loss: float = INF
    decisions: list[LayerDecision] | None = None
    reason: str = ""

    def rates(self, n_layers: int | None = None) -> list[float]:
        if not self.decisions:
            return []
        if n_layers is None:
            n_layers = max(d.layer for d in self.decisions) + 1
        rates = [0.0] * n_layers
        for decision in self.decisions:
            rates[decision.layer] = decision.rate
        return rates

    def timeouts(self, n_layers: int | None = None) -> list[float]:
        if not self.decisions:
            return []
        if n_layers is None:
            n_layers = max(d.layer for d in self.decisions) + 1
        timeouts = [0.0] * n_layers
        for decision in self.decisions:
            timeouts[decision.layer] = decision.timeout_ms
        return timeouts


@dataclass
class JointScheduleResult:
    mode: str
    feasible: bool
    split_m: int | None
    local: LocalScheduleResult | None
    tx_ms: float = 0.0
    end_ms: float = 0.0
    total_ms: float = INF
    offloaded_layers: list[int] | None = None
    reason: str = ""


def _time_bucket(ms: float, quantum_ms: float) -> int:
    return int(math.ceil(ms / quantum_ms - 1e-12))


def _bucket_ms(bucket: int, quantum_ms: float) -> float:
    return bucket * quantum_ms


def load_profile(path: Path) -> tuple[list[LayerProfile], dict[str, Any]]:
    data = json.loads(path.read_text())
    if "core_splits" in data:
        splits, meta = load_core_split_profile(path)
        if not splits:
            return [], meta
        # Backward-compatible view: expose the first split to old callers.
        return list(splits[0].layers), meta
    layers = []
    for layer_obj in data["layers"]:
        candidates = tuple(
            Candidate(
                rate=float(c["rate"]),
                main_ms=float(c["main_ms"]),
                loss=float(c.get("loss", 0.0)),
                weight=float(c.get("weight", c["rate"])),
                tail_ms=float(c.get("tail_ms", 0.0)),
            )
            for c in layer_obj["candidates"]
        )
        layers.append(LayerProfile(layer=int(layer_obj["layer"]), candidates=candidates))
    layers.sort(key=lambda item: item.layer)
    meta = {k: v for k, v in data.items() if k != "layers"}
    return layers, meta


def _parse_cpus(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        cpus: list[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                cpus.extend(range(int(lo_s), int(hi_s) + 1))
            else:
                cpus.append(int(part))
        return tuple(sorted(set(cpus)))
    return tuple(int(x) for x in value)


def _load_layers(layer_objs: Iterable[dict[str, Any]]) -> tuple[LayerProfile, ...]:
    layers = []
    for layer_obj in layer_objs:
        candidates = tuple(
            Candidate(
                rate=float(c["rate"]),
                main_ms=float(c["main_ms"]),
                loss=float(c.get("loss", 0.0)),
                weight=float(c.get("weight", c["rate"])),
                tail_ms=float(c.get("tail_ms", 0.0)),
            )
            for c in layer_obj["candidates"]
        )
        layers.append(LayerProfile(layer=int(layer_obj["layer"]), candidates=candidates))
    return tuple(sorted(layers, key=lambda item: item.layer))


def load_core_split_profile(path: Path) -> tuple[list[CoreSplitProfile], dict[str, Any]]:
    data = json.loads(path.read_text())
    if "core_splits" not in data:
        layers, meta = load_profile(path)
        return [
            CoreSplitProfile(
                split_id="legacy",
                p=int(meta.get("p", 0)),
                major_cpus=_parse_cpus(meta.get("major_cpus")),
                minor_cpus=_parse_cpus(meta.get("minor_cpus")),
                layers=tuple(layers),
            )
        ], meta

    splits = []
    for index, split_obj in enumerate(data["core_splits"]):
        major_cpus = _parse_cpus(split_obj.get("major_cpus"))
        minor_cpus = _parse_cpus(split_obj.get("minor_cpus"))
        split_id = str(split_obj.get("split_id") or f"p{len(major_cpus)}_{index}")
        splits.append(
            CoreSplitProfile(
                split_id=split_id,
                p=int(split_obj.get("p", len(major_cpus))),
                major_cpus=major_cpus,
                minor_cpus=minor_cpus,
                layers=_load_layers(split_obj["layers"]),
            )
        )
    meta = {k: v for k, v in data.items() if k != "core_splits"}
    return splits, meta


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def solve_local_dp(
    layers: Iterable[LayerProfile],
    deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
    split: CoreSplitProfile | None = None,
) -> LocalScheduleResult:
    """Solve chapter 5.2 DP with adjacent-clipped-layer exclusion."""

    layer_list = list(layers)
    if not layer_list:
        return LocalScheduleResult(
            feasible=True,
            deadline_ms=deadline_ms,
            split_id=split.split_id if split else None,
            p=split.p if split else None,
            major_cpus=list(split.major_cpus) if split else None,
            minor_cpus=list(split.minor_cpus) if split else None,
            total_main_ms=0.0,
            total_loss=0.0,
            decisions=[],
        )

    max_bucket = _time_bucket(deadline_ms, quantum_ms)
    # state: (time_bucket, current_clipped) -> (loss, prev_key, candidate)
    states: dict[tuple[int, int], tuple[float, tuple[int, int] | None, Candidate | None]] = {
        (0, 0): (0.0, None, None)
    }
    parents: list[dict[tuple[int, int], tuple[tuple[int, int], Candidate]]] = []

    for layer in layer_list:
        next_states: dict[tuple[int, int], tuple[float, tuple[int, int] | None, Candidate | None]] = {}
        layer_parent: dict[tuple[int, int], tuple[tuple[int, int], Candidate]] = {}
        for prev_key, (prev_loss, _, _) in states.items():
            prev_bucket, prev_clipped = prev_key
            for candidate in layer.candidates:
                clipped = candidate.clipped
                if prev_clipped + clipped > 1:
                    continue
                bucket = prev_bucket + _time_bucket(candidate.main_ms, quantum_ms)
                if bucket > max_bucket:
                    continue
                key = (bucket, clipped)
                loss = prev_loss + candidate.loss
                old = next_states.get(key)
                if old is None or loss < old[0] - 1e-12:
                    next_states[key] = (loss, prev_key, candidate)
                    layer_parent[key] = (prev_key, candidate)
        parents.append(layer_parent)
        states = next_states
        if not states:
            return LocalScheduleResult(
                feasible=False,
                deadline_ms=deadline_ms,
                split_id=split.split_id if split else None,
                p=split.p if split else None,
                major_cpus=list(split.major_cpus) if split else None,
                minor_cpus=list(split.minor_cpus) if split else None,
                reason=f"no state survives after layer {layer.layer}",
            )

    best_key: tuple[int, int] | None = None
    best_loss = INF
    best_bucket = math.inf
    for key, (loss, _, _) in states.items():
        bucket, _ = key
        if loss < best_loss - 1e-12 or (abs(loss - best_loss) <= 1e-12 and bucket < best_bucket):
            best_key = key
            best_loss = loss
            best_bucket = bucket

    if best_key is None:
        return LocalScheduleResult(
            feasible=False,
            deadline_ms=deadline_ms,
            split_id=split.split_id if split else None,
            p=split.p if split else None,
            major_cpus=list(split.major_cpus) if split else None,
            minor_cpus=list(split.minor_cpus) if split else None,
            reason="no terminal state",
        )

    decisions_rev: list[LayerDecision] = []
    key = best_key
    for layer, layer_parent in zip(reversed(layer_list), reversed(parents)):
        prev_key, candidate = layer_parent[key]
        decisions_rev.append(
            LayerDecision(
                layer=layer.layer,
                rate=candidate.rate,
                main_ms=candidate.main_ms,
                loss=candidate.loss,
                weight=candidate.weight,
                tail_ms=candidate.tail_ms,
            )
        )
        key = prev_key
    decisions = list(reversed(decisions_rev))
    allocate_timeouts(decisions, timeout_budget_ms)
    return LocalScheduleResult(
        feasible=True,
        deadline_ms=deadline_ms,
        split_id=split.split_id if split else None,
        p=split.p if split else None,
        major_cpus=list(split.major_cpus) if split else None,
        minor_cpus=list(split.minor_cpus) if split else None,
        total_main_ms=sum(item.main_ms for item in decisions),
        total_loss=sum(item.loss for item in decisions),
        decisions=decisions,
    )


def allocate_timeouts(decisions: list[LayerDecision], timeout_budget_ms: float) -> None:
    total_weight = sum(max(0.0, item.weight) for item in decisions)
    if timeout_budget_ms <= 0.0 or total_weight <= 0.0:
        for item in decisions:
            item.timeout_ms = 0.0
        return
    for item in decisions:
        item.timeout_ms = timeout_budget_ms * max(0.0, item.weight) / total_weight


def _metric_series(meta: dict[str, Any], name: str, n_layers: int) -> list[float]:
    value = meta.get(name)
    if value is None:
        return [0.0] * (n_layers + 1)
    if isinstance(value, (int, float)):
        return [float(value)] * (n_layers + 1)
    series = [float(x) for x in value]
    if len(series) < n_layers + 1:
        series.extend([series[-1] if series else 0.0] * (n_layers + 1 - len(series)))
    return series


def _nonzero_rate_count(result: LocalScheduleResult | None) -> int:
    if result is None or not result.decisions:
        return 0
    return sum(1 for item in result.decisions if item.rate > 0.0)


def _local_choice_key(result: LocalScheduleResult) -> tuple[float, int, int, float]:
    # The DP objective is minimum loss. Ties should naturally prefer the
    # no-SVD/no-minor case when all cores are idle and full local execution fits.
    p = result.p if result.p is not None else 0
    return (
        result.total_loss,
        _nonzero_rate_count(result),
        -p,
        result.total_main_ms,
    )


def solve_local_dp_over_splits(
    splits: Iterable[CoreSplitProfile],
    deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
) -> LocalScheduleResult:
    best: LocalScheduleResult | None = None
    failures = []
    for split in splits:
        result = solve_local_dp(
            split.layers,
            deadline_ms=deadline_ms,
            timeout_budget_ms=timeout_budget_ms,
            quantum_ms=quantum_ms,
            split=split,
        )
        if result.feasible:
            if best is None or _local_choice_key(result) < _local_choice_key(best):
                best = result
        else:
            failures.append(f"{split.split_id}: {result.reason}")
    if best is not None:
        return best
    return LocalScheduleResult(
        feasible=False,
        deadline_ms=deadline_ms,
        reason="; ".join(failures) if failures else "no core split candidates",
    )


def solve_joint_schedule(
    layers: list[LayerProfile],
    local_deadline_ms: float,
    request_deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> JointScheduleResult:
    """Solve chapter 5.4 flow: local DP first, then suffix offload search."""

    meta = meta or {}
    local = solve_local_dp(layers, local_deadline_ms, timeout_budget_ms, quantum_ms)
    n_layers = len(layers)
    if local.feasible:
        return JointScheduleResult(
            mode="local",
            feasible=True,
            split_m=n_layers - 1,
            local=local,
            total_ms=local.total_main_ms,
            offloaded_layers=[],
        )

    tx_series = _metric_series(meta, "tx_ms_by_split_m", n_layers)
    end_series = _metric_series(meta, "end_ms_by_split_m", n_layers)

    for split_m in range(n_layers - 1, -1, -1):
        tx_ms = tx_series[split_m]
        end_ms = end_series[split_m]
        edge_deadline = request_deadline_ms - tx_ms - end_ms
        if edge_deadline < 0.0:
            continue
        prefix = layers[: split_m + 1]
        edge = solve_local_dp(prefix, edge_deadline, timeout_budget_ms, quantum_ms)
        if not edge.feasible:
            continue
        total_ms = edge.total_main_ms + tx_ms + end_ms
        if total_ms <= request_deadline_ms + 1e-9:
            return JointScheduleResult(
                mode="edge_end",
                feasible=True,
                split_m=split_m,
                local=edge,
                tx_ms=tx_ms,
                end_ms=end_ms,
                total_ms=total_ms,
                offloaded_layers=list(range(split_m + 1, n_layers)),
            )

    return JointScheduleResult(
        mode="none",
        feasible=False,
        split_m=None,
        local=local,
        reason="local DP infeasible and no suffix offload split satisfies the request deadline",
    )


def solve_joint_schedule_over_splits(
    splits: list[CoreSplitProfile],
    local_deadline_ms: float,
    request_deadline_ms: float,
    timeout_budget_ms: float = 0.0,
    quantum_ms: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> JointScheduleResult:
    """Solve Algorithmv2 5.2 with outer core-split enumeration.

    The choice is lexicographic:
    1. prefer any feasible local-only plan over offload;
    2. among local plans, minimize loss, then clipping count, then prefer
       larger p, then lower main time;
    3. if local is infeasible, choose the suffix offload plan with the fewest
       offloaded layers, then lower loss, then lower total time.
    """

    meta = meta or {}
    if not splits:
        return JointScheduleResult(
            mode="none",
            feasible=False,
            split_m=None,
            local=None,
            reason="no core split candidates",
        )

    local = solve_local_dp_over_splits(
        splits,
        deadline_ms=local_deadline_ms,
        timeout_budget_ms=timeout_budget_ms,
        quantum_ms=quantum_ms,
    )
    n_layers = len(splits[0].layers)
    if local.feasible:
        return JointScheduleResult(
            mode="local",
            feasible=True,
            split_m=n_layers - 1,
            local=local,
            total_ms=local.total_main_ms,
            offloaded_layers=[],
        )

    tx_series = _metric_series(meta, "tx_ms_by_split_m", n_layers)
    end_series = _metric_series(meta, "end_ms_by_split_m", n_layers)
    best: JointScheduleResult | None = None
    best_key: tuple[int, float, float, int] | None = None

    for split_m in range(n_layers - 1, -1, -1):
        tx_ms = tx_series[split_m]
        end_ms = end_series[split_m]
        edge_deadline = request_deadline_ms - tx_ms - end_ms
        if edge_deadline < 0.0:
            continue

        prefix_splits = [
            CoreSplitProfile(
                split_id=split.split_id,
                p=split.p,
                major_cpus=split.major_cpus,
                minor_cpus=split.minor_cpus,
                layers=tuple(layer for layer in split.layers if layer.layer <= split_m),
            )
            for split in splits
        ]
        edge = solve_local_dp_over_splits(
            prefix_splits,
            deadline_ms=edge_deadline,
            timeout_budget_ms=timeout_budget_ms,
            quantum_ms=quantum_ms,
        )
        if not edge.feasible:
            continue
        total_ms = edge.total_main_ms + tx_ms + end_ms
        if total_ms > request_deadline_ms + 1e-9:
            continue
        offloaded = n_layers - split_m - 1
        key = (offloaded, edge.total_loss, total_ms, _nonzero_rate_count(edge))
        if best_key is None or key < best_key:
            best_key = key
            best = JointScheduleResult(
                mode="edge_end",
                feasible=True,
                split_m=split_m,
                local=edge,
                tx_ms=tx_ms,
                end_ms=end_ms,
                total_ms=total_ms,
                offloaded_layers=list(range(split_m + 1, n_layers)),
            )

    if best is not None:
        return best

    return JointScheduleResult(
        mode="none",
        feasible=False,
        split_m=None,
        local=local,
        reason="local DP infeasible and no suffix offload split satisfies the request deadline",
    )


def write_rate_file(path: Path, rates: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(f"{max(0.0, min(1.0, rate)):.6g}" for rate in rates) + "\n")


def write_timeout_file(path: Path, timeouts_ms: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(f"{max(0.0, timeout):.6g}" for timeout in timeouts_ms) + "\n")


def result_to_json(result: JointScheduleResult, n_layers: int) -> dict[str, Any]:
    obj = asdict(result)
    if result.local and result.local.decisions is not None:
        obj["rates"] = result.local.rates(n_layers)
        obj["timeouts_ms"] = result.local.timeouts(n_layers)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve exp12 load-aware SVD schedules.")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--local-deadline-ms", required=True, type=float)
    parser.add_argument("--request-deadline-ms", type=float)
    parser.add_argument("--timeout-budget-ms", type=float, default=0.0)
    parser.add_argument("--quantum-ms", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-rates", type=Path)
    parser.add_argument("--out-timeouts", type=Path)
    args = parser.parse_args()

    splits, meta = load_core_split_profile(args.profile)
    request_deadline_ms = args.request_deadline_ms
    if request_deadline_ms is None:
        request_deadline_ms = args.local_deadline_ms

    result = solve_joint_schedule_over_splits(
        splits=splits,
        local_deadline_ms=args.local_deadline_ms,
        request_deadline_ms=request_deadline_ms,
        timeout_budget_ms=args.timeout_budget_ms,
        quantum_ms=args.quantum_ms,
        meta=meta,
    )
    n_layers = len(splits[0].layers) if splits else 0
    obj = result_to_json(result, n_layers)
    print(json.dumps(obj, ensure_ascii=False, indent=2))

    if args.out_json:
        save_json(args.out_json, obj)
    if args.out_rates and result.local is not None and result.local.decisions is not None:
        write_rate_file(args.out_rates, result.local.rates(n_layers))
    if args.out_timeouts and result.local is not None and result.local.decisions is not None:
        write_timeout_file(args.out_timeouts, result.local.timeouts(n_layers))
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
