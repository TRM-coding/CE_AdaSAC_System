import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GGUF_PY_ROOT = ROOT / "3dparty" / "llamacpp" / "gguf-py"
if str(GGUF_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(GGUF_PY_ROOT))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
from gguf import GGUFWriter


DEFAULT_REPO = "microsoft/resnet-50"


def set_proxy(proxy_url: str | None) -> None:
    if not proxy_url:
        return
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["ALL_PROXY"] = proxy_url


def maybe_download(repo_id: str, model_dir: Path, proxy_url: str | None) -> Path:
    set_proxy(proxy_url)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ],
    )
    return model_dir


def fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight.detach().float()
    if conv.bias is None:
        bias = torch.zeros(weight.shape[0], dtype=torch.float32)
    else:
        bias = conv.bias.detach().float()

    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mean = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    inv_std = torch.rsqrt(var + bn.eps)

    scale = gamma * inv_std
    fused_weight = weight * scale.view(-1, 1, 1, 1)
    fused_bias = beta + (bias - mean) * scale
    return fused_weight, fused_bias


def to_ggml_conv_storage(weight: torch.Tensor) -> np.ndarray:
    # Keep the NumPy buffer in [out, in, kh, kw] C-order so that the raw bytes
    # match ggml's dim0-fastest interpretation for a tensor with ne=[kw, kh, in, out].
    return weight.contiguous().cpu().numpy().astype(np.float32)


def to_bias_4d(bias: torch.Tensor) -> np.ndarray:
    return bias.view(1, 1, -1, 1).contiguous().cpu().numpy().astype(np.float32)


def conv_weight_to_matrix(weight: torch.Tensor) -> np.ndarray:
    # [out, in, kh, kw] -> 2D matrix with shape [out, in * kh * kw].
    return weight.reshape(weight.shape[0], -1).contiguous().cpu().numpy().astype(np.float32)


def svd_factorize_conv_weight(
    weight: torch.Tensor,
    rank_ratio: float,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = conv_weight_to_matrix(weight)
    matrix_torch = torch.from_numpy(matrix)
    u, s, vh = torch.linalg.svd(matrix_torch, full_matrices=False)

    full_rank = s.shape[0]
    keep_rank = max(1, min(full_rank, int(round(full_rank * rank_ratio))))

    u = u[:, :keep_rank]
    s = s[:keep_rank]
    vh = vh[:keep_rank, :]

    s_sqrt = torch.sqrt(s)
    u_factor = (u * s_sqrt.unsqueeze(0)).contiguous().cpu().numpy().astype(dtype)
    v_factor = (s_sqrt.unsqueeze(1) * vh).contiguous().cpu().numpy().astype(dtype)
    return u_factor, v_factor


def add_conv_bn(
    writer: GGUFWriter,
    prefix: str,
    conv: torch.nn.Conv2d,
    bn: torch.nn.BatchNorm2d,
    rank_ratio: float,
    svd_dtype: np.dtype,
) -> None:
    fused_weight, fused_bias = fuse_conv_bn(conv, bn)
    conv_weight = to_ggml_conv_storage(fused_weight)
    conv_bias = to_bias_4d(fused_bias)
    svd_u, svd_v = svd_factorize_conv_weight(fused_weight, rank_ratio, svd_dtype)
    writer.add_tensor(f"{prefix}.weight", conv_weight, raw_shape=tuple(conv_weight.shape))
    writer.add_tensor(f"{prefix}.bias", conv_bias, raw_shape=tuple(conv_bias.shape[::-1]))
    writer.add_tensor(f"{prefix}.weight_svd_u", svd_u, raw_shape=tuple(svd_u.shape))
    writer.add_tensor(f"{prefix}.weight_svd_v", svd_v, raw_shape=tuple(svd_v.shape))


def ordered_labels(config) -> list[str]:
    labels: list[str] = []
    for idx in range(config.num_labels):
        key = idx if idx in config.id2label else str(idx)
        labels.append(str(config.id2label[key]))
    return labels


def convert(
    model_dir: Path,
    output_path: Path,
    repo_id: str,
    conv_svd_rank_ratio: float,
    conv_svd_dtype: str,
) -> None:
    model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval()
    processor = AutoImageProcessor.from_pretrained(str(model_dir))

    if not (0.0 < conv_svd_rank_ratio <= 1.0):
        raise ValueError("--conv-svd-rank-ratio must be in (0, 1]")

    if conv_svd_dtype == "f16":
        svd_dtype = np.float16
    elif conv_svd_dtype == "f32":
        svd_dtype = np.float32
    else:
        raise ValueError(f"unsupported --conv-svd-dtype: {conv_svd_dtype}")

    writer = GGUFWriter(str(output_path), "resnet50")
    writer.add_name("resnet50")
    writer.add_description("ResNet-50 image classification model with Conv+BN fused for ggml inference")
    writer.add_author("OpenAI Codex")
    writer.add_string("general.source.huggingface.repository", repo_id)

    resize_shorter = round(processor.size["shortest_edge"] / processor.crop_pct)
    image_size = processor.size["shortest_edge"]

    writer.add_uint32("resnet50.image_size", int(image_size))
    writer.add_uint32("resnet50.resize_shorter_edge", int(resize_shorter))
    writer.add_uint32("resnet50.num_classes", int(model.config.num_labels))
    writer.add_array("resnet50.stage_block_count", [3, 4, 6, 3])
    writer.add_array("resnet50.input_mean", [float(x) for x in processor.image_mean])
    writer.add_array("resnet50.input_std", [float(x) for x in processor.image_std])
    writer.add_array("resnet50.labels", ordered_labels(model.config))

    stem = model.resnet.embedder.embedder
    add_conv_bn(
        writer,
        "resnet.stem.conv",
        stem.convolution,
        stem.normalization,
        conv_svd_rank_ratio,
        svd_dtype,
    )

    for stage_idx, stage in enumerate(model.resnet.encoder.stages):
        for block_idx, block in enumerate(stage.layers):
            base = f"resnet.stage.{stage_idx}.block.{block_idx}"

            if hasattr(block.shortcut, "convolution"):
                add_conv_bn(
                    writer,
                    f"{base}.downsample",
                    block.shortcut.convolution,
                    block.shortcut.normalization,
                    conv_svd_rank_ratio,
                    svd_dtype,
                )

            conv1 = block.layer[0]
            conv2 = block.layer[1]
            conv3 = block.layer[2]

            add_conv_bn(writer, f"{base}.conv1", conv1.convolution, conv1.normalization, conv_svd_rank_ratio, svd_dtype)
            add_conv_bn(writer, f"{base}.conv2", conv2.convolution, conv2.normalization, conv_svd_rank_ratio, svd_dtype)
            add_conv_bn(writer, f"{base}.conv3", conv3.convolution, conv3.normalization, conv_svd_rank_ratio, svd_dtype)

    fc = model.classifier[1]
    writer.add_tensor(
        "resnet.classifier.weight",
        fc.weight.detach().float().contiguous().cpu().numpy().astype(np.float32),
        raw_shape=tuple(fc.weight.detach().float().shape),
    )
    writer.add_tensor(
        "resnet.classifier.bias",
        fc.bias.detach().float().view(-1, 1).contiguous().cpu().numpy().astype(np.float32),
        raw_shape=tuple(fc.bias.detach().float().view(-1, 1).shape[::-1]),
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Hugging Face ResNet-50 to GGUF for ggml inference")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--model-dir", default="/tmp/resnet50_hf/model")
    parser.add_argument("--outfile", default=str(ROOT / "gguf_models" / "resnet50-f32.gguf"))
    parser.add_argument("--proxy-url", default=os.environ.get("https_proxy") or os.environ.get("http_proxy"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--conv-svd-rank-ratio", type=float, default=1.0)
    parser.add_argument("--conv-svd-dtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        maybe_download(args.repo_id, model_dir, args.proxy_url)

    convert(
        model_dir,
        output_path,
        args.repo_id,
        args.conv_svd_rank_ratio,
        args.conv_svd_dtype,
    )
    print(f"saved GGUF to: {output_path}")


if __name__ == "__main__":
    main()
