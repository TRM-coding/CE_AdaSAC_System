import argparse
import re

import numpy as np
import torch

from gguf import GGMLQuantizationType, GGUFValueType
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf.quants import dequantize


NEED_SVD = ("ffn_up", "ffn_down", "ffn_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qwen2_svd GGUF from a dense or quantized GGUF model.")
    parser.add_argument("input", help="input GGUF path")
    parser.add_argument("output", nargs="?", help="output GGUF path, defaults to <input>.sort_svd.gguf")
    parser.add_argument(
        "--device",
        default="cuda:5" if torch.cuda.is_available() and torch.cuda.device_count() > 5 else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="torch device used for SVD",
    )
    return parser.parse_args()


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    for key_in, field in reader.fields.items():
        if key_in in ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]:
            continue

        key_out = key_in if "rope" in key_in else key_in.replace("qwen2", "qwen2_svd")
        ftype = field.types[0]

        if ftype == GGUFValueType.STRING:
            value = "".join(chr(i) for i in field.parts[field.data[0]])
            writer.add_string(key=key_out, val=value)
        elif ftype == GGUFValueType.ARRAY:
            writer.add_array(key=key_out, val=field.contents())
        else:
            value = field.parts[field.data[0]][0]
            writer.add_key_value(key=key_out, val=value, vtype=ftype)


def tensor_to_float32_matrix(tensor) -> np.ndarray:
    raw = np.array(tensor.data, copy=False)
    shape = tuple(int(x) for x in tensor.shape[::-1])

    if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        matrix = raw.astype(np.float32, copy=False).reshape(shape)
    else:
        matrix = dequantize(raw, tensor.tensor_type).reshape(shape)

    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)

    return np.ascontiguousarray(matrix)


def svd_factorize_torch(matrix: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    weight = torch.from_numpy(matrix).to(device)
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)

    s_sqrt = torch.sqrt(s)
    u_factor = (u * s_sqrt.unsqueeze(0)).to("cpu", dtype=torch.float16).numpy()
    v_factor = (s_sqrt.unsqueeze(1) * vh).to("cpu", dtype=torch.float16).numpy()
    return u_factor, v_factor


def svd_tensor_names(name: str) -> tuple[str, str]:
    if "ffn_down" in name:
        return re.sub(r"ffn_down", r"ffn_down_svd_u", name), re.sub(r"ffn_down", r"ffn_down_svd_v", name)
    if "ffn_up" in name:
        return re.sub(r"ffn_up", r"ffn_up_svd_u", name), re.sub(r"ffn_up", r"ffn_up_svd_v", name)
    if "ffn_gate" in name:
        return re.sub(r"ffn_gate", r"ffn_gate_svd_u", name), re.sub(r"ffn_gate", r"ffn_gate_svd_v", name)
    raise ValueError(f"unsupported SVD tensor name: {name}")


def main() -> int:
    args = parse_args()
    input_path = args.input
    output_path = args.output or f"{input_path}.sort_svd.gguf"

    reader = GGUFReader(input_path)
    writer = GGUFWriter(output_path, "qwen2_svd")
    copy_metadata(reader, writer)

    svd_count = 0
    for tensor in reader.tensors:
        if any(token in tensor.name for token in NEED_SVD):
            matrix = tensor_to_float32_matrix(tensor)
            u_factor, v_factor = svd_factorize_torch(matrix, args.device)
            out_u, out_v = svd_tensor_names(tensor.name)

            writer.add_tensor(
                name=out_u,
                tensor=u_factor,
                raw_shape=tuple(u_factor.shape),
                raw_dtype=GGMLQuantizationType.F16,
            )
            writer.add_tensor(
                name=out_v,
                tensor=v_factor,
                raw_shape=tuple(v_factor.shape),
                raw_dtype=GGMLQuantizationType.F16,
            )
            svd_count += 1

        writer.add_tensor(
            name=tensor.name,
            tensor=tensor.data,
            raw_shape=tuple(tensor.shape)[::-1],
            raw_dtype=tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"device: {args.device}")
    print(f"svd tensors: {svd_count}")
    print("FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
