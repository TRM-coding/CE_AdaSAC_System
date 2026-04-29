#!/usr/bin/env python3
"""Rerun the exp6 Q4 direct SVD truncation speed experiment on isolated CPUs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager


REPO = Path("/home/tianruiming/CE_ADA_LLAMA")
LLAMA = REPO / "src/llama.cpp"
OUT_DIR = LLAMA / "datas/svd_effect"
DEFAULT_BINARY = REPO / "build-release-current/decode_svd_test"
DEFAULT_LIB = REPO / "build-release-current/bin"
DEFAULT_MODEL = LLAMA / "gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf"
CHINESE_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
]


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
    return ",".join(str(cpu) for cpu in cpus)


def make_cgroup(name: str, cpus: list[int]) -> Path:
    cg = Path("/sys/fs/cgroup") / name
    commands = [
        f"mkdir -p {shlex.quote(str(cg))}",
        "echo +cpuset > /sys/fs/cgroup/cgroup.subtree_control || true",
        f"echo 0 > {shlex.quote(str(cg / 'cpuset.mems'))}",
        f"echo {shlex.quote(cpu_spec(cpus))} > {shlex.quote(str(cg / 'cpuset.cpus'))}",
    ]
    subprocess.run(["sudo", "-n", "bash", "-lc", "; ".join(commands)], check=True)
    return cg


def run_in_cgroup(cmd: list[str], cg: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    script = (
        f"echo $$ > {shlex.quote(str(cg / 'cgroup.procs'))}; "
        f"export LD_LIBRARY_PATH={shlex.quote(str(DEFAULT_LIB))}; "
        f"exec {shlex.join(cmd)}"
    )
    return subprocess.run(
        ["sudo", "-n", "bash", "-lc", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        timeout=timeout_s,
    )


def write_rates(path: Path, rate: float, n_layers: int = 28) -> None:
    path.write_text(",".join(f"{rate:.6g}" for _ in range(n_layers)) + "\n")


def parse_output(text: str) -> dict[str, Any]:
    def f(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "decode_tps": f(r"Decode-only throughput:\s*([0-9.]+)\s*tokens/s"),
        "e2e_tps": f(r"End-to-end throughput:\s*([0-9.]+)\s*tokens/s"),
        "decode_sec": f(r"tokens in ([0-9.]+)\s*s of llama_decode"),
        "generation_decode_ms": f(r"generation_decode=([0-9.]+)\s*ms"),
        "ffn_svd_total_ms": f(r"ffn_svd_total=([0-9.]+)\s*ms"),
        "generated_text": (re.search(r"Generated text:\s*(.*)", text) or [None, None])[1],
        "succeed": "SUCCEED" in text,
        "local_truncation_enabled": "SVD local truncation enabled" in text,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rate: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_rate.setdefault(row["rate"], []).append(row)
    base_vals = [row["decode_tps"] for row in by_rate[0.0] if row["decode_tps"] is not None]
    base = statistics.fmean(base_vals)
    out = []
    for rate in sorted(by_rate):
        vals = [row["decode_tps"] for row in by_rate[rate] if row["decode_tps"] is not None]
        e2e = [row["e2e_tps"] for row in by_rate[rate] if row["e2e_tps"] is not None]
        dec = [row["decode_sec"] for row in by_rate[rate] if row["decode_sec"] is not None]
        avg = statistics.fmean(vals)
        out.append({
            "truncation_rate": rate,
            "kept_rank_fraction": 1.0 - rate,
            "runs": len(vals),
            "decode_tps_mean": avg,
            "decode_tps_stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "decode_tps_min": min(vals),
            "decode_tps_max": max(vals),
            "e2e_tps_mean": statistics.fmean(e2e) if e2e else None,
            "decode_sec_mean": statistics.fmean(dec) if dec else None,
            "speedup_vs_full_rank": avg / base,
        })
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fin:
        rows = list(csv.DictReader(fin))
    numeric_keys = {
        "truncation_rate",
        "kept_rank_fraction",
        "runs",
        "decode_tps_mean",
        "decode_tps_stdev",
        "decode_tps_min",
        "decode_tps_max",
        "e2e_tps_mean",
        "decode_sec_mean",
        "speedup_vs_full_rank",
    }
    for row in rows:
        for key in numeric_keys:
            if key in row and row[key] not in ("", None):
                row[key] = float(row[key])
        if "runs" in row:
            row["runs"] = int(row["runs"])
    return rows


def plot(summary: list[dict[str, Any]], output_prefix: str) -> None:
    chinese_font = None
    for font_path in CHINESE_FONT_CANDIDATES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            chinese_font = font_manager.FontProperties(fname=str(font_path)).get_name()
            break

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            chinese_font or "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "Droid Sans Fallback",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    rates = [row["truncation_rate"] for row in summary]
    speedups = [row["speedup_vs_full_rank"] for row in summary]
    tps = [row["decode_tps_mean"] for row in summary]
    err = [row["decode_tps_stdev"] for row in summary]

    fig, ax1 = plt.subplots(figsize=(5.9, 3.45))
    ax1.plot(rates, speedups, marker="o", linewidth=2.2, color="#1f77b4", label="加速比")
    ax1.axhline(1.0, color="#8c8c8c", linewidth=1.0, linestyle="--")
    ax1.set_xlabel("SVD 截断率")
    ax1.set_ylabel("相对完整秩的加速比", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(rates)
    ax1.set_xticklabels([f"{rate:.1f}" for rate in rates])
    ax1.grid(axis="y", color="#dddddd", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.errorbar(rates, tps, yerr=err, marker="s", linewidth=1.8, capsize=3, color="#4d4d4d", label="解码吞吐")
    ax2.set_ylabel("解码吞吐（tokens/s）", color="#4d4d4d")
    ax2.tick_params(axis="y", labelcolor="#4d4d4d")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{output_prefix}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cpus", default="60-67")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rates", default="0,0.5,0.7,0.8")
    parser.add_argument("--cgroup-name", default="svd_effect_q4_repro")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--split-a", default="off")
    parser.add_argument("--split-b", default="off")
    parser.add_argument("--tail-mode", default="")
    parser.add_argument("--output-prefix", default="q4_direct_truncation_rerun")
    parser.add_argument("--plot-only-summary", type=Path)
    args = parser.parse_args()

    if args.plot_only_summary is not None:
        plot(read_summary_csv(args.plot_only_summary), args.output_prefix)
        return 0

    cpus = parse_cpu_list(args.cpus)
    cg = make_cgroup(args.cgroup_name, cpus)
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    rates_dir = OUT_DIR / "repro_rates"
    logs_dir = OUT_DIR / f"{args.output_prefix}_logs"
    rates_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    for rate in rates:
        write_rates(rates_dir / f"rate_{rate:.1f}.txt", rate)

    rows: list[dict[str, Any]] = []
    for rate in rates:
        for rep in range(1, args.repeats + 1):
            rate_arg = "off" if rate == 0.0 else str(rates_dir / f"rate_{rate:.1f}.txt")
            cmd = [
                "taskset", "-c", cpu_spec(cpus),
                str(args.binary), str(args.model),
                str(args.tokens), str(args.threads), "0",
                "off", rate_arg,
                args.split_a, args.split_b, "0.5", "0", "2", "off",
            ]
            if args.tail_mode:
                cmd.append(args.tail_mode)
            completed = run_in_cgroup(cmd, cg, args.timeout_s)
            log_path = logs_dir / f"rate_{rate:.1f}_rep{rep}.log"
            log_path.write_text(completed.stdout)
            parsed = parse_output(completed.stdout)
            row = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rate": rate,
                "repeat": rep,
                "returncode": completed.returncode,
                "log": str(log_path.relative_to(OUT_DIR)),
                **parsed,
            }
            rows.append(row)
            print(f"rate={rate:.1f} rep={rep} tok/s={row['decode_tps']} trunc={row['local_truncation_enabled']} rc={completed.returncode}", flush=True)

    summary = summarize(rows)
    (OUT_DIR / f"{args.output_prefix}_raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (OUT_DIR / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_csv(OUT_DIR / f"{args.output_prefix}_summary.csv", summary)
    plot(summary, args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
