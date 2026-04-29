#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def run_timed(cmd: list[str], cwd: Path, repeats: int) -> dict:
    times: list[float] = []
    last_stdout = ""
    for i in range(repeats + 1):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)
        elapsed = time.perf_counter() - t0
        last_stdout = proc.stdout
        if i > 0:
            times.append(elapsed)
    return {
        "times_sec": times,
        "mean_sec": statistics.mean(times),
        "median_sec": statistics.median(times),
        "stdev_sec": statistics.stdev(times) if len(times) > 1 else 0.0,
        "top1": last_stdout.strip().splitlines()[0] if last_stdout.strip() else "",
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Benchmark ResNet50 conv SVD fold vs im2col paths")
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--svd-bin", default="./build-release-current/run_resnet50_conv_svd")
    parser.add_argument("--baseline-bin", default="./build-release-current/run_resnet50")
    parser.add_argument("--svd-model", required=True)
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--threads", default="8")
    parser.add_argument("--top-k", default="5")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    common = ["--image", args.image, "--threads", args.threads, "--top-k", args.top_k]
    results = {
        "im2col_svd": run_timed(
            [args.svd_bin, "--model", args.svd_model, *common, "--mode", "im2col"],
            args.repo,
            args.repeats,
        ),
        "fold_svd": run_timed(
            [args.svd_bin, "--model", args.svd_model, *common, "--mode", "fold"],
            args.repo,
            args.repeats,
        ),
    }

    if args.baseline_model:
        results["baseline_conv"] = run_timed(
            [args.baseline_bin, "--model", args.baseline_model, *common],
            args.repo,
            args.repeats,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
