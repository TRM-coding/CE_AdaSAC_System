#!/usr/bin/env python3
"""Generate CE-AdaSAC-style synthetic mini-batches for ResNet50 and Qwen2.5."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoModelForImageClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_DATA_DIR = EXP_DIR / "data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def processor_size(processor: Any) -> int:
    size = getattr(processor, "size", 224)
    if isinstance(size, dict):
        return int(size.get("shortest_edge") or size.get("height") or size.get("width") or 224)
    return int(size)


def image_norm_stats(processor: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), device=device).view(1, 3, 1, 1)
    std = torch.tensor(getattr(processor, "image_std", [0.229, 0.224, 0.225]), device=device).view(1, 3, 1, 1)
    return mean, std


def target_classes(num_classes: int, batch_size: int) -> torch.Tensor:
    if batch_size <= num_classes:
        return torch.linspace(0, num_classes - 1, steps=batch_size).round().long()
    return torch.arange(batch_size).remainder(num_classes).long()


def generate_resnet50(args: argparse.Namespace) -> Path:
    model_dir = Path(args.resnet_model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    image_size = processor_size(processor)
    mean, std = image_norm_stats(processor, device)
    targets = target_classes(model.config.num_labels, args.resnet_batch_size).to(device)

    raw = torch.randn(args.resnet_batch_size, 3, image_size, image_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=args.resnet_lr)
    history: list[dict[str, float]] = []
    started = time.time()

    for step in range(args.resnet_steps):
        pixels_01 = raw.sigmoid()
        pixel_values = (pixels_01 - mean) / std
        logits = model(pixel_values=pixel_values).logits
        ce_loss = F.cross_entropy(logits, targets)
        tv_loss = (
            (pixels_01[:, :, 1:, :] - pixels_01[:, :, :-1, :]).abs().mean()
            + (pixels_01[:, :, :, 1:] - pixels_01[:, :, :, :-1]).abs().mean()
        )
        loss = ce_loss + args.resnet_tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.resnet_steps:
            probs = logits.softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "cross_entropy": float(ce_loss.detach().cpu()),
                    "target_prob_mean": float(probs.mean().detach().cpu()),
                    "target_prob_min": float(probs.min().detach().cpu()),
                }
            )

    with torch.no_grad():
        pixels_01 = raw.sigmoid()
        pixel_values = (pixels_01 - mean) / std
        logits = model(pixel_values=pixel_values).logits
        probs = logits.softmax(dim=-1)
        target_probs = probs.gather(1, targets[:, None]).squeeze(1)
        top_probs, top_ids = probs.topk(k=min(5, probs.shape[-1]), dim=-1)

    labels = [str(model.config.id2label.get(int(t), int(t))) for t in targets.detach().cpu()]
    payload = {
        "pixel_values": pixel_values.detach().cpu().to(torch.float16),
        "images_uint8": (pixels_01.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8),
        "target_class_ids": targets.detach().cpu(),
        "target_labels": labels,
        "target_probs": target_probs.detach().cpu().to(torch.float32),
        "top5_class_ids": top_ids.detach().cpu(),
        "top5_probs": top_probs.detach().cpu().to(torch.float32),
        "metadata": {
            "method": "CE-AdaSAC synthetic data generator: optimize input by cross-entropy to target class",
            "model_dir": str(model_dir),
            "image_size": image_size,
            "steps": args.resnet_steps,
            "lr": args.resnet_lr,
            "tv_weight": args.resnet_tv_weight,
            "seed": args.seed,
            "elapsed_sec": time.time() - started,
        },
        "history": history,
    }

    out_path = out_dir / "resnet50_synthetic_minibatch.pt"
    torch.save(payload, out_path)
    save_json(
        out_dir / "resnet50_synthetic_minibatch_manifest.json",
        {
            "file": out_path.name,
            "batch_size": args.resnet_batch_size,
            "target_class_ids": [int(x) for x in targets.detach().cpu()],
            "target_labels": labels,
            "target_prob_mean": float(target_probs.mean().detach().cpu()),
            "target_prob_min": float(target_probs.min().detach().cpu()),
            "history": history,
            "metadata": payload["metadata"],
        },
    )
    return out_path


def qwen_target_tokens(tokenizer: Any, batch_size: int) -> torch.Tensor:
    seed_texts = [
        " the",
        " answer",
        " data",
        " model",
        " cloud",
        " edge",
        " inference",
        " compression",
        " accuracy",
        " latency",
        " 中国",
        " 测试",
        " 数据",
        " 模型",
        " 推理",
        " 优化",
    ]
    ids: list[int] = []
    for text in seed_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    if not ids:
        ids = [int(tokenizer.eos_token_id or 0)]
    while len(ids) < batch_size:
        ids.extend(ids)
    return torch.tensor(ids[:batch_size], dtype=torch.long)


def nearest_tokens(input_embeds: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    embeds = F.normalize(input_embeds.float(), dim=-1)
    vocab = F.normalize(embed_weight.float(), dim=-1)
    nearest: list[torch.Tensor] = []
    for row in embeds:
        nearest.append(torch.matmul(row, vocab.T).argmax(dim=-1).cpu())
    return torch.stack(nearest, dim=0)


def generate_qwen(args: argparse.Namespace) -> Path:
    model_dir = Path(args.qwen_model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if args.qwen_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach()
    targets = qwen_target_tokens(tokenizer, args.qwen_batch_size).to(device)

    random_ids = torch.randint(
        low=0,
        high=int(embed_weight.shape[0]),
        size=(args.qwen_batch_size, args.qwen_seq_len),
        device=device,
    )
    init_embeds = embed_layer(random_ids).detach().float()
    init_embeds = init_embeds + args.qwen_init_noise * torch.randn_like(init_embeds)
    input_embeds = torch.nn.Parameter(init_embeds)
    attention_mask = torch.ones(args.qwen_batch_size, args.qwen_seq_len, dtype=torch.long, device=device)
    position_ids = torch.arange(args.qwen_seq_len, device=device).view(1, -1).expand(args.qwen_batch_size, -1)
    optimizer = torch.optim.Adam([input_embeds], lr=args.qwen_lr)
    history: list[dict[str, float]] = []
    started = time.time()

    for step in range(args.qwen_steps):
        outputs = model(inputs_embeds=input_embeds.to(model.dtype), attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits[:, -1, :].float()
        ce_loss = F.cross_entropy(logits, targets)
        norm_loss = input_embeds.float().pow(2).mean()
        smooth_loss = (input_embeds[:, 1:, :] - input_embeds[:, :-1, :]).pow(2).mean()
        loss = ce_loss + args.qwen_l2_weight * norm_loss + args.qwen_smooth_weight * smooth_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.qwen_steps:
            probs = logits.softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "cross_entropy": float(ce_loss.detach().cpu()),
                    "target_prob_mean": float(probs.mean().detach().cpu()),
                    "target_prob_min": float(probs.min().detach().cpu()),
                }
            )

    with torch.no_grad():
        outputs = model(inputs_embeds=input_embeds.to(model.dtype), attention_mask=attention_mask, position_ids=position_ids)
        logits = outputs.logits[:, -1, :].float()
        probs = logits.softmax(dim=-1)
        target_probs = probs.gather(1, targets[:, None]).squeeze(1)
        top_probs, top_ids = probs.topk(k=5, dim=-1)
        projected_ids = nearest_tokens(input_embeds.detach().cpu(), embed_weight.detach().cpu())

    target_texts = tokenizer.batch_decode(targets.detach().cpu().view(-1, 1), skip_special_tokens=False)
    projected_texts = tokenizer.batch_decode(projected_ids, skip_special_tokens=False)
    payload = {
        "inputs_embeds": input_embeds.detach().cpu().to(torch.float16),
        "nearest_input_ids": projected_ids.to(torch.long),
        "attention_mask": attention_mask.detach().cpu(),
        "position_ids": position_ids.detach().cpu(),
        "target_token_ids": targets.detach().cpu(),
        "target_texts": target_texts,
        "projected_texts": projected_texts,
        "target_probs": target_probs.detach().cpu().to(torch.float32),
        "top5_token_ids": top_ids.detach().cpu(),
        "top5_probs": top_probs.detach().cpu().to(torch.float32),
        "metadata": {
            "method": "CE-AdaSAC synthetic data generator: optimize continuous transformer inputs by cross-entropy to target vocabulary id",
            "model_dir": str(model_dir),
            "seq_len": args.qwen_seq_len,
            "steps": args.qwen_steps,
            "lr": args.qwen_lr,
            "l2_weight": args.qwen_l2_weight,
            "smooth_weight": args.qwen_smooth_weight,
            "seed": args.seed,
            "elapsed_sec": time.time() - started,
            "note": "inputs_embeds are the primary synthetic inputs; nearest_input_ids are a lossy token projection for token-only pipelines.",
        },
        "history": history,
    }

    out_path = out_dir / "qwen2_5_1_5b_synthetic_minibatch.pt"
    torch.save(payload, out_path)
    save_json(
        out_dir / "qwen2_5_1_5b_synthetic_minibatch_manifest.json",
        {
            "file": out_path.name,
            "batch_size": args.qwen_batch_size,
            "seq_len": args.qwen_seq_len,
            "target_token_ids": [int(x) for x in targets.detach().cpu()],
            "target_texts": target_texts,
            "projected_texts": projected_texts,
            "target_prob_mean": float(target_probs.mean().detach().cpu()),
            "target_prob_min": float(target_probs.min().detach().cpu()),
            "history": history,
            "metadata": payload["metadata"],
        },
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CE-AdaSAC synthetic mini-batch test data.")
    parser.add_argument("--task", choices=["all", "resnet50", "qwen"], default="all")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--resnet-model-dir", default=str(DEFAULT_MODELS_DIR / "resnet50_hf_model"))
    parser.add_argument("--qwen-model-dir", default=str(DEFAULT_MODELS_DIR / "qwen2_5_1_5b"))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--resnet-batch-size", type=int, default=8)
    parser.add_argument("--resnet-steps", type=int, default=80)
    parser.add_argument("--resnet-lr", type=float, default=0.08)
    parser.add_argument("--resnet-tv-weight", type=float, default=1e-4)

    parser.add_argument("--qwen-batch-size", type=int, default=4)
    parser.add_argument("--qwen-seq-len", type=int, default=16)
    parser.add_argument("--qwen-steps", type=int, default=30)
    parser.add_argument("--qwen-lr", type=float, default=0.08)
    parser.add_argument("--qwen-init-noise", type=float, default=0.01)
    parser.add_argument("--qwen-l2-weight", type=float, default=0.0)
    parser.add_argument("--qwen-smooth-weight", type=float, default=0.0)
    parser.add_argument("--qwen-gradient-checkpointing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    outputs: list[Path] = []
    if args.task in {"all", "resnet50"}:
        outputs.append(generate_resnet50(args))
    if args.task in {"all", "qwen"}:
        outputs.append(generate_qwen(args))

    for path in outputs:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
