#!/usr/bin/env python3
"""Compare synthetic mini-batches against equal-sized real batches under SVD pruning.

The experiment intentionally uses a local, non-cooperative SVD semantics:
selected Linear/Conv2d modules are replaced by low-rank factors and the omitted
tail singular directions are not evaluated elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoModelForImageClassification, AutoTokenizer


ROOT = Path("/home/tianruiming/CE_ADA_LLAMA")
LLAMA = ROOT / "src/llama.cpp"
EXP11_DATA = LLAMA / "exp11_make_input_data/data"
DEFAULT_QWEN_MODEL = LLAMA / "models/qwen2_5_1_5b"
DEFAULT_RESNET_MODEL = LLAMA / "models/resnet50_hf_model"
DEFAULT_QWEN_SYN = EXP11_DATA / "qwen2_5_1_5b_synthetic_minibatch.pt"
DEFAULT_RESNET_SYN = EXP11_DATA / "resnet50_synthetic_minibatch.pt"
DEFAULT_QWEN_REAL_TEXT = LLAMA / "exp10_svd_local_truncation_fix/ppl_corpus_qwen_out_64k.txt"
DEFAULT_IMAGENET_VAL = Path("/SSD/val")
DEFAULT_IMAGENET_DEVKIT = Path("/SSD/ImageNet/ILSVRC2012_devkit_t12.tar.gz")


@dataclass(frozen=True)
class PrunePlan:
    policy_id: int
    module_to_keep: dict[str, float]
    pruned_params: int
    total_params: int

    @property
    def prune_fraction(self) -> float:
        return self.pruned_params / self.total_params if self.total_params else 0.0


@dataclass
class CachedLinearSVD:
    u_scaled: torch.Tensor
    vh: torch.Tensor
    bias: torch.Tensor | None
    rank: int
    param_count: int


@dataclass
class CachedConvSVD:
    u_scaled: torch.Tensor
    vh: torch.Tensor
    bias: torch.Tensor | None
    rank: int
    param_count: int
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]


CachedSVD = CachedLinearSVD | CachedConvSVD


class LowRankLinear(nn.Module):
    def __init__(self, module: nn.Linear, keep_ratio: float, svd_device: torch.device):
        super().__init__()
        weight = module.weight.detach().to(svd_device, dtype=torch.float32)
        rank = min(weight.shape)
        keep = max(1, min(rank, math.ceil(rank * keep_ratio)))
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        self.v = nn.Parameter(vh[:keep, :].to(module.weight.device, dtype=module.weight.dtype), requires_grad=False)
        self.u = nn.Parameter((u[:, :keep] * s[:keep]).to(module.weight.device, dtype=module.weight.dtype), requires_grad=False)
        self.bias = None
        if module.bias is not None:
            self.bias = nn.Parameter(module.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x, self.v)
        return F.linear(hidden, self.u, self.bias)


class CachedLowRankLinear(nn.Module):
    def __init__(self, cached: CachedLinearSVD, keep: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.v = nn.Parameter(cached.vh[:keep, :].to(device=device, dtype=dtype), requires_grad=False)
        self.u = nn.Parameter(cached.u_scaled[:, :keep].to(device=device, dtype=dtype), requires_grad=False)
        self.bias = None
        if cached.bias is not None:
            self.bias = nn.Parameter(cached.bias.to(device=device, dtype=dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x, self.v)
        return F.linear(hidden, self.u, self.bias)


class LowRankConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d, keep_ratio: float, svd_device: torch.device):
        super().__init__()
        if module.groups != 1:
            raise ValueError("LowRankConv2d only supports groups=1")
        out_ch, in_ch, kh, kw = module.weight.shape
        flat = module.weight.detach().reshape(out_ch, in_ch * kh * kw).to(svd_device, dtype=torch.float32)
        rank = min(flat.shape)
        keep = max(1, min(rank, math.ceil(rank * keep_ratio)))
        u, s, vh = torch.linalg.svd(flat, full_matrices=False)
        first = vh[:keep, :].reshape(keep, in_ch, kh, kw).to(module.weight.device, dtype=module.weight.dtype)
        second = (u[:, :keep] * s[:keep]).reshape(out_ch, keep, 1, 1).to(module.weight.device, dtype=module.weight.dtype)
        self.first = nn.Parameter(first, requires_grad=False)
        self.second = nn.Parameter(second, requires_grad=False)
        self.bias = None
        if module.bias is not None:
            self.bias = nn.Parameter(module.bias.detach().clone(), requires_grad=False)
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.conv2d(x, self.first, None, self.stride, self.padding, self.dilation, 1)
        return F.conv2d(hidden, self.second, self.bias, 1, 0, 1, 1)


class CachedLowRankConv2d(nn.Module):
    def __init__(self, cached: CachedConvSVD, keep: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        kh, kw = cached.kernel_size
        first = cached.vh[:keep, :].reshape(keep, cached.in_channels, kh, kw)
        second = cached.u_scaled[:, :keep].reshape(cached.out_channels, keep, 1, 1)
        self.first = nn.Parameter(first.to(device=device, dtype=dtype), requires_grad=False)
        self.second = nn.Parameter(second.to(device=device, dtype=dtype), requires_grad=False)
        self.bias = None
        if cached.bias is not None:
            self.bias = nn.Parameter(cached.bias.to(device=device, dtype=dtype), requires_grad=False)
        self.stride = cached.stride
        self.padding = cached.padding
        self.dilation = cached.dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.conv2d(x, self.first, None, self.stride, self.padding, self.dilation, 1)
        return F.conv2d(hidden, self.second, self.bias, 1, 0, 1, 1)


def set_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def linear_param_count(module: nn.Linear) -> int:
    return int(module.weight.numel())


def conv_param_count(module: nn.Conv2d) -> int:
    return int(module.weight.numel())


def rank_pruned_params(module: nn.Module, keep_ratio: float) -> int:
    if isinstance(module, nn.Linear):
        rank = min(module.weight.shape)
        keep = max(1, min(rank, math.ceil(rank * keep_ratio)))
        return int(module.weight.numel() * (rank - keep) / rank)
    if isinstance(module, nn.Conv2d):
        out_ch, in_ch, kh, kw = module.weight.shape
        rank = min(out_ch, in_ch * kh * kw)
        keep = max(1, min(rank, math.ceil(rank * keep_ratio)))
        return int(module.weight.numel() * (rank - keep) / rank)
    raise TypeError(type(module))


def module_rank_and_params(module: nn.Module) -> tuple[int, int]:
    if isinstance(module, nn.Linear):
        return min(module.weight.shape), int(module.weight.numel())
    if isinstance(module, nn.Conv2d):
        out_ch, in_ch, kh, kw = module.weight.shape
        return min(out_ch, in_ch * kh * kw), int(module.weight.numel())
    raise TypeError(type(module))


def keep_ratio_to_keep(rank: int, keep_ratio: float) -> int:
    return max(1, min(rank, math.ceil(rank * keep_ratio)))


def keep_to_pruned_params(param_count: int, rank: int, keep: int) -> int:
    return int(round(param_count * (rank - keep) / rank))


def named_target_modules(model: nn.Module, task: str, qwen_patterns: tuple[str, ...]) -> list[tuple[str, nn.Module]]:
    out: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if task == "qwen":
            if isinstance(module, nn.Linear) and any(pattern in name for pattern in qwen_patterns):
                out.append((name, module))
        elif task == "resnet50":
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                out.append((name, module))
        else:
            raise ValueError(task)
    return out


def make_targeted_prune_plans(
    modules: list[tuple[str, nn.Module]],
    num_policies: int,
    min_prune_percent: float,
    max_prune_percent: float,
    seed: int,
) -> list[PrunePlan]:
    rng = random.Random(seed)
    specs = []
    total = 0
    max_prunable = 0
    for name, module in modules:
        rank, params = module_rank_and_params(module)
        cap = keep_to_pruned_params(params, rank, 1)
        specs.append((name, rank, params, cap))
        total += params
        max_prunable += cap

    plans: list[PrunePlan] = [PrunePlan(0, {}, 0, total)]
    if num_policies <= 0:
        return plans

    if num_policies == 1:
        targets = [max_prune_percent]
    else:
        step = (max_prune_percent - min_prune_percent) / (num_policies - 1)
        targets = [min_prune_percent + i * step for i in range(num_policies)]

    for policy_id, target_percent in enumerate(targets, start=1):
        budget = min(max_prunable, int(round(total * target_percent / 100.0)))
        ordered = specs[:]
        rng.shuffle(ordered)
        caps_suffix = [0] * (len(ordered) + 1)
        for i in range(len(ordered) - 1, -1, -1):
            caps_suffix[i] = caps_suffix[i + 1] + ordered[i][3]

        keep_by_name: dict[str, float] = {}
        actual_pruned = 0
        remaining = budget
        for i, (name, rank, params, cap) in enumerate(ordered):
            if remaining <= 0:
                break
            rest_cap = caps_suffix[i + 1]
            min_take = max(0, remaining - rest_cap)
            max_take = min(cap, remaining)
            if max_take <= 0:
                continue
            take = rng.randint(min_take, max_take) if max_take > min_take else max_take
            keep = max(1, min(rank, round(rank * (1.0 - take / params))))
            pruned = keep_to_pruned_params(params, rank, keep)
            if pruned == 0 and take > 0 and rank > 1:
                keep = rank - 1
                pruned = keep_to_pruned_params(params, rank, keep)
            keep_by_name[name] = keep / rank
            actual_pruned += pruned
            remaining = max(0, budget - actual_pruned)

        plans.append(PrunePlan(policy_id, keep_by_name, actual_pruned, total))
    return plans


def make_prune_plans(
    modules: list[tuple[str, nn.Module]],
    num_policies: int,
    keep_min: float,
    keep_max: float,
    max_modules_per_policy: int | None,
    seed: int,
) -> list[PrunePlan]:
    rng = random.Random(seed)
    total = sum(int(m.weight.numel()) for _, m in modules if isinstance(m, (nn.Linear, nn.Conv2d)))
    plans: list[PrunePlan] = [PrunePlan(0, {}, 0, total)]
    module_names = [name for name, _ in modules]
    module_by_name = dict(modules)
    for policy_id in range(1, num_policies + 1):
        selected = module_names[:]
        rng.shuffle(selected)
        if max_modules_per_policy is not None:
            selected = selected[:max_modules_per_policy]
        ratios = {name: rng.uniform(keep_min, keep_max) for name in selected}
        pruned = sum(rank_pruned_params(module_by_name[name], ratio) for name, ratio in ratios.items())
        plans.append(PrunePlan(policy_id, ratios, pruned, total))
    return plans


def build_resnet_svd_cache(
    modules: list[tuple[str, nn.Module]],
    svd_device: torch.device,
    cache_device: torch.device,
) -> dict[str, CachedSVD]:
    cache: dict[str, CachedSVD] = {}
    for name, module in modules:
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().to(svd_device, dtype=torch.float32)
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            cache[name] = CachedLinearSVD(
                u_scaled=(u * s).to(cache_device),
                vh=vh.to(cache_device),
                bias=module.bias.detach().to(cache_device) if module.bias is not None else None,
                rank=min(weight.shape),
                param_count=int(module.weight.numel()),
            )
        elif isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise ValueError(f"cached SVD only supports groups=1: {name}")
            out_ch, in_ch, kh, kw = module.weight.shape
            flat = module.weight.detach().reshape(out_ch, in_ch * kh * kw).to(svd_device, dtype=torch.float32)
            u, s, vh = torch.linalg.svd(flat, full_matrices=False)
            cache[name] = CachedConvSVD(
                u_scaled=(u * s).to(cache_device),
                vh=vh.to(cache_device),
                bias=module.bias.detach().to(cache_device) if module.bias is not None else None,
                rank=min(flat.shape),
                param_count=int(module.weight.numel()),
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(kh, kw),
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
            )
        else:
            raise TypeError(type(module))
    return cache


def apply_plan(model: nn.Module, plan: PrunePlan, task: str, svd_device: torch.device) -> None:
    modules = dict(model.named_modules())
    for name, keep_ratio in plan.module_to_keep.items():
        old = modules[name]
        if isinstance(old, nn.Linear):
            set_module(model, name, LowRankLinear(old, keep_ratio, svd_device))
        elif isinstance(old, nn.Conv2d):
            set_module(model, name, LowRankConv2d(old, keep_ratio, svd_device))
        else:
            raise TypeError(f"Unsupported module in {task}: {name}: {type(old)}")


def apply_cached_resnet_plan(
    model: nn.Module,
    plan: PrunePlan,
    cache: dict[str, CachedSVD],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for name, keep_ratio in plan.module_to_keep.items():
        cached = cache[name]
        keep = keep_ratio_to_keep(cached.rank, keep_ratio)
        if isinstance(cached, CachedLinearSVD):
            set_module(model, name, CachedLowRankLinear(cached, keep, device, dtype))
        elif isinstance(cached, CachedConvSVD):
            set_module(model, name, CachedLowRankConv2d(cached, keep, device, dtype))
        else:
            raise TypeError(type(cached))


def qwen_real_batch(tokenizer, text_path: Path, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    needed = batch_size * (seq_len + 1)
    if token_ids.numel() < needed:
        repeats = math.ceil(needed / max(1, token_ids.numel()))
        token_ids = token_ids.repeat(repeats)
    token_ids = token_ids[:needed].reshape(batch_size, seq_len + 1)
    return {
        "input_ids": token_ids[:, :-1].contiguous(),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "target_token_ids": token_ids[:, -1].contiguous(),
    }


@torch.no_grad()
def qwen_last_token_ppl(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> float:
    if "inputs_embeds" in batch:
        outputs = model(
            inputs_embeds=batch["inputs_embeds"].to(device=device, dtype=dtype),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch.get("position_ids", None).to(device) if "position_ids" in batch else None,
        )
    else:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
    logits = outputs.logits[:, -1, :].float()
    targets = batch["target_token_ids"].to(device)
    return float(torch.exp(F.cross_entropy(logits, targets)).detach().cpu())


def load_imagenet_synset_to_class(devkit: Path, id2label: dict[int, str] | None = None) -> dict[str, int]:
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy is required to map ImageNet synset folders to class ids") from exc

    with tarfile.open(devkit, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("meta.mat"))
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"cannot read meta.mat from {devkit}")
        meta = loadmat(extracted, squeeze_me=True)["synsets"]
    label_to_id = {label: idx for idx, label in (id2label or {}).items()}
    mapping: dict[str, int] = {}
    for row in meta:
        ilsvrc_id = int(row["ILSVRC2012_ID"])
        wnid = str(row["WNID"])
        words = str(row["words"])
        if words in label_to_id:
            mapping[wnid] = label_to_id[words]
        elif 1 <= ilsvrc_id <= 1000:
            mapping[wnid] = ilsvrc_id - 1
    return mapping


def load_imagenet_batch(
    val_dir: Path,
    devkit: Path,
    batch_size: int,
    image_size: int,
    seed: int,
    id2label: dict[int, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    synset_to_class = load_imagenet_synset_to_class(devkit, id2label)
    files = sorted(p for p in val_dir.glob("*/*.JPEG") if p.parent.name in synset_to_class)
    if len(files) < batch_size:
        raise RuntimeError(f"not enough ImageNet val images under {val_dir}")
    rng = random.Random(seed)
    selected = rng.sample(files, batch_size)
    preproc = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    images = []
    labels = []
    for path in selected:
        with Image.open(path) as img:
            images.append(preproc(img.convert("RGB")))
        labels.append(synset_to_class[path.parent.name])
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def resnet_acc(model: nn.Module, pixel_values: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    logits = model(pixel_values=pixel_values.to(device=device, dtype=next(model.parameters()).dtype).float()).logits
    pred = logits.argmax(dim=-1).detach().cpu()
    return float((pred == labels.cpu()).float().mean().item())


@torch.no_grad()
def resnet_pair_acc(
    model: nn.Module,
    syn_pixels: torch.Tensor,
    syn_labels: torch.Tensor,
    real_pixels: torch.Tensor,
    real_labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    pixels = torch.cat([syn_pixels, real_pixels], dim=0).to(device=device).float()
    logits = model(pixel_values=pixels).logits
    pred = logits.argmax(dim=-1).detach().cpu()
    n_syn = syn_labels.numel()
    syn_acc = float((pred[:n_syn] == syn_labels.cpu()).float().mean().item())
    real_acc = float((pred[n_syn:] == real_labels.cpu()).float().mean().item())
    return syn_acc, real_acc


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(csv_path: Path, output_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, task, metric in [(axes[0], "qwen", "ppl"), (axes[1], "resnet50", "acc")]:
        sub = df[df["task"] == task].sort_values("prune_fraction")
        if sub.empty:
            ax.set_visible(False)
            continue
        for source, marker in [("synthetic", "o"), ("real", "s")]:
            part = sub[sub["source"] == source]
            x = part["prune_fraction"] * 100.0
            y = part[metric]
            if task == "resnet50":
                ax.plot(x, y, linewidth=1.3, alpha=0.45)
                ax.scatter(x, y, marker=marker, s=44, alpha=0.86, label=source)
            else:
                ax.plot(x, y, marker=marker, label=source)
        ax.set_xlabel("Total SVD pruning amount (%)")
        ax.set_ylabel("PPL" if metric == "ppl" else "Top-1 ACC")
        ax.set_title("Qwen2.5-1.5B" if task == "qwen" else "ResNet50")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_qwen(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    syn = torch.load(args.qwen_synthetic, map_location="cpu")
    batch_size, seq_len = syn["inputs_embeds"].shape[:2]
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, local_files_only=True, trust_remote_code=True)
    real = qwen_real_batch(tokenizer, args.qwen_real_text, batch_size, seq_len)

    probe = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    targets = named_target_modules(probe, "qwen", tuple(args.qwen_module_patterns.split(",")))
    plans = make_prune_plans(
        targets, args.num_policies, args.keep_min, args.keep_max, args.max_qwen_modules_per_policy, args.seed
    )
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for plan in plans:
        model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model, local_files_only=True, trust_remote_code=True, torch_dtype=dtype
        ).to(device).eval()
        if plan.module_to_keep:
            apply_plan(model, plan, "qwen", torch.device(args.svd_device))
        syn_ppl = qwen_last_token_ppl(model, syn, device, dtype)
        real_ppl = qwen_last_token_ppl(model, real, device, dtype)
        for source, ppl in [("synthetic", syn_ppl), ("real", real_ppl)]:
            rows.append(
                {
                    "task": "qwen",
                    "policy_id": plan.policy_id,
                    "source": source,
                    "prune_fraction": plan.prune_fraction,
                    "ppl": ppl,
                    "acc": "",
                    "num_pruned_modules": len(plan.module_to_keep),
                    "plan_json": json.dumps(plan.module_to_keep, sort_keys=True),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_resnet(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    device = torch.device(args.device)
    dtype = torch.float32
    syn = torch.load(args.resnet_synthetic, map_location="cpu")
    syn_pixels = syn["pixel_values"].float()
    syn_labels = syn["target_class_ids"].long()
    probe = AutoModelForImageClassification.from_pretrained(args.resnet_model, local_files_only=True).to(device).eval()
    id2label = {int(k): v for k, v in probe.config.id2label.items()}
    real_pixels, real_labels = load_imagenet_batch(
        args.imagenet_val, args.imagenet_devkit, syn_pixels.shape[0], syn_pixels.shape[-1], args.seed, id2label
    )
    targets = named_target_modules(probe, "resnet50", ())
    if args.resnet_target_prune_sweep:
        plans = make_targeted_prune_plans(
            targets,
            args.num_policies,
            args.resnet_min_prune_percent,
            args.resnet_max_prune_percent,
            args.seed + 17,
        )
    else:
        plans = make_prune_plans(
            targets, args.num_policies, args.keep_min, args.keep_max, args.max_resnet_modules_per_policy, args.seed + 17
        )
    svd_device = torch.device(args.svd_device)
    cache_device = device if args.resnet_cache_svd_on_device else torch.device("cpu")
    svd_cache = build_resnet_svd_cache(targets, svd_device, cache_device)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for plan in plans:
        model = AutoModelForImageClassification.from_pretrained(args.resnet_model, local_files_only=True).to(device).eval()
        if plan.module_to_keep:
            apply_cached_resnet_plan(model, plan, svd_cache, device, dtype)
        syn_acc, real_acc = resnet_pair_acc(model, syn_pixels, syn_labels, real_pixels, real_labels, device)
        for source, acc in [("synthetic", syn_acc), ("real", real_acc)]:
            rows.append(
                {
                    "task": "resnet50",
                    "policy_id": plan.policy_id,
                    "source": source,
                    "prune_fraction": plan.prune_fraction,
                    "ppl": "",
                    "acc": acc,
                    "num_pruned_modules": len(plan.module_to_keep),
                    "plan_json": json.dumps(plan.module_to_keep, sort_keys=True),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="qwen,resnet50", help="comma-separated subset: qwen,resnet50")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results/minibatch_vs_real_svd")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--svd-device", default="cpu", help="where to run torch.linalg.svd; cpu saves GPU memory")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-policies", type=int, default=5)
    parser.add_argument("--keep-min", type=float, default=0.45)
    parser.add_argument("--keep-max", type=float, default=0.9)

    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-synthetic", type=Path, default=DEFAULT_QWEN_SYN)
    parser.add_argument("--qwen-real-text", type=Path, default=DEFAULT_QWEN_REAL_TEXT)
    parser.add_argument("--qwen-module-patterns", default="mlp.gate_proj,mlp.up_proj,mlp.down_proj")
    parser.add_argument("--max-qwen-modules-per-policy", type=int, default=6)

    parser.add_argument("--resnet-model", type=Path, default=DEFAULT_RESNET_MODEL)
    parser.add_argument("--resnet-synthetic", type=Path, default=DEFAULT_RESNET_SYN)
    parser.add_argument("--imagenet-val", type=Path, default=DEFAULT_IMAGENET_VAL)
    parser.add_argument("--imagenet-devkit", type=Path, default=DEFAULT_IMAGENET_DEVKIT)
    parser.add_argument("--max-resnet-modules-per-policy", type=int, default=10)
    parser.add_argument(
        "--resnet-target-prune-sweep",
        action="store_true",
        help="for ResNet50, generate randomized plans with target total pruning percentages",
    )
    parser.add_argument("--resnet-min-prune-percent", type=float, default=1.0)
    parser.add_argument("--resnet-max-prune-percent", type=float, default=100.0)
    parser.add_argument(
        "--resnet-cache-svd-on-device",
        action="store_true",
        help="keep cached ResNet SVD factors on --device after computing them; faster on CUDA, more VRAM",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < args.keep_min <= args.keep_max <= 1.0):
        raise ValueError("--keep-min/--keep-max must satisfy 0 < min <= max <= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    torch.manual_seed(args.seed)
    if "qwen" in tasks:
        run_qwen(args, rows)
    if "resnet50" in tasks:
        run_resnet(args, rows)
    csv_path = args.out_dir / "svd_minibatch_vs_real.csv"
    png_path = args.out_dir / "svd_minibatch_vs_real.png"
    write_rows(csv_path, rows)
    plot_results(csv_path, png_path)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
