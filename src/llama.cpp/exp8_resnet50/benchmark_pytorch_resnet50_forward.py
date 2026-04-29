#!/usr/bin/env python3
import argparse
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch ResNet50 forward only")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    model_dir = Path(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval()
    processor = AutoImageProcessor.from_pretrained(str(model_dir))

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    logits = None
    with torch.inference_mode():
        for _ in range(args.warmup):
            logits = model(**inputs).logits

        times_ms: list[float] = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            logits = model(**inputs).logits
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    assert logits is not None
    top = torch.topk(logits[0], k=5)
    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    print(
        f"forward_ms_mean={mean_ms}"
        f"\tforward_ms_median={median_ms}"
        f"\tforward_ms_min={min(times_ms)}"
        f"\tforward_ms_max={max(times_ms)}"
        f"\trepeats={len(times_ms)}"
    )
    for rank, (idx, score) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
        label = model.config.id2label.get(idx, model.config.id2label.get(str(idx), "<unknown>"))
        print(f"{rank}\tclass_id={idx}\tlogit={score}\tlabel={label}")


if __name__ == "__main__":
    main()
