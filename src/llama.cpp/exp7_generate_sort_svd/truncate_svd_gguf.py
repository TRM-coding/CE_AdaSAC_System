import argparse
from pathlib import Path

import numpy as np

from gguf import GGMLQuantizationType, GGUFValueType
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truncate SVD ranks in a qwen2_svd GGUF model.")
    parser.add_argument("input", help="input qwen2_svd GGUF path")
    parser.add_argument("output", help="output GGUF path")
    parser.add_argument(
        "--keep-ratio",
        type=float,
        required=True,
        help="fraction of singular components to keep in each SVD factor pair, in (0, 1].",
    )
    parser.add_argument(
        "--rank-align",
        type=int,
        default=1,
        help="round kept rank down to a multiple of this value; use 32 for Q4_0-friendly shapes.",
    )
    return parser.parse_args()


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    for key, field in reader.fields.items():
        if key in ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]:
            continue

        ftype = field.types[0]
        if ftype == GGUFValueType.STRING:
            value = "".join(chr(i) for i in field.parts[field.data[0]])
            writer.add_string(key=key, val=value)
        elif ftype == GGUFValueType.ARRAY:
            writer.add_array(key=key, val=field.contents())
        else:
            value = field.parts[field.data[0]][0]
            writer.add_key_value(key=key, val=value, vtype=ftype)


def tensor_to_array(tensor) -> np.ndarray:
    shape = tuple(int(x) for x in tensor.shape[::-1])
    dtype = np.float32 if tensor.tensor_type == GGMLQuantizationType.F32 else np.float16
    return np.array(tensor.data, copy=True).astype(dtype, copy=False).reshape(shape)


def truncate_rank_length(length: int, keep_ratio: float, rank_align: int) -> int:
    keep = int(np.ceil(length * keep_ratio))
    keep = max(1, min(length, keep))
    if rank_align > 1:
        aligned = (keep // rank_align) * rank_align
        if aligned == 0:
            aligned = rank_align if rank_align <= length else length
        keep = min(length, aligned)
    return keep


def main() -> int:
    args = parse_args()
    keep_ratio = args.keep_ratio
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"--keep-ratio must be in (0, 1], got {keep_ratio}")
    if args.rank_align < 1:
        raise ValueError(f"--rank-align must be >= 1, got {args.rank_align}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    reader = GGUFReader(str(input_path))
    writer = GGUFWriter(str(output_path), "qwen2_svd")
    copy_metadata(reader, writer)

    truncated = 0
    for tensor in reader.tensors:
        name = tensor.name
        if name.endswith("_svd_u.weight") or name.endswith("_svd_v.weight"):
            if tensor.tensor_type not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                raise ValueError(f"expected F16/F32 SVD factor, got {tensor.tensor_type} for {name}")

            array = tensor_to_array(tensor)
            rank_dim = array.shape[1] if name.endswith("_svd_u.weight") else array.shape[0]
            keep = truncate_rank_length(rank_dim, keep_ratio, args.rank_align)
            array = array[:, :keep] if name.endswith("_svd_u.weight") else array[:keep, :]

            writer.add_tensor(
                name=name,
                tensor=np.ascontiguousarray(array),
                raw_shape=tuple(array.shape),
                raw_dtype=tensor.tensor_type,
            )
            truncated += 1
            continue

        writer.add_tensor(
            name=name,
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
    print(f"keep_ratio: {keep_ratio}")
    print(f"rank_align: {args.rank_align}")
    print(f"truncated tensors: {truncated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
