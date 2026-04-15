import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/3dparty/llamacpp/gguf-py")

from gguf import GGMLQuantizationType, GGUFValueType, GGUFWriter, LlamaFileType
from gguf.gguf_reader import GGUFReader
from gguf.quants import quantize


SVD_TENSOR_TOKENS = (
    "ffn_up_svd_u",
    "ffn_up_svd_v",
    "ffn_gate_svd_u",
    "ffn_gate_svd_v",
    "ffn_down_svd_u",
    "ffn_down_svd_v",
)

QTYPE_ALIASES = {
    "f16": GGMLQuantizationType.F16,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q2_k": GGMLQuantizationType.Q2_K,
    "q3_k": GGMLQuantizationType.Q3_K,
    "q4_k": GGMLQuantizationType.Q4_K,
    "q5_k": GGMLQuantizationType.Q5_K,
    "q6_k": GGMLQuantizationType.Q6_K,
    "iq4_nl": GGMLQuantizationType.IQ4_NL,
    "iq4_xs": GGMLQuantizationType.IQ4_XS,
}

FILE_TYPE_ALIASES = {
    GGMLQuantizationType.F16: LlamaFileType.MOSTLY_F16,
    GGMLQuantizationType.Q4_0: LlamaFileType.MOSTLY_Q4_0,
    GGMLQuantizationType.Q4_1: LlamaFileType.MOSTLY_Q4_1,
    GGMLQuantizationType.Q5_0: LlamaFileType.MOSTLY_Q5_0,
    GGMLQuantizationType.Q5_1: LlamaFileType.MOSTLY_Q5_1,
    GGMLQuantizationType.Q8_0: LlamaFileType.MOSTLY_Q8_0,
    GGMLQuantizationType.Q2_K: LlamaFileType.MOSTLY_Q2_K,
    GGMLQuantizationType.Q3_K: LlamaFileType.MOSTLY_Q3_K_M,
    GGMLQuantizationType.Q4_K: LlamaFileType.MOSTLY_Q4_K_M,
    GGMLQuantizationType.Q5_K: LlamaFileType.MOSTLY_Q5_K_M,
    GGMLQuantizationType.Q6_K: LlamaFileType.MOSTLY_Q6_K,
    GGMLQuantizationType.IQ4_NL: LlamaFileType.MOSTLY_IQ4_NL,
    GGMLQuantizationType.IQ4_XS: LlamaFileType.MOSTLY_IQ4_XS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-quantize SVD tensors for a mobile-side qwen2_svd GGUF model.",
    )
    parser.add_argument("input", help="input compact SVD GGUF path")
    parser.add_argument(
        "output",
        nargs="?",
        help="output GGUF path, defaults to <input>.mobile_<quant>.gguf",
    )
    parser.add_argument(
        "--quant",
        default="q8_0",
        choices=sorted(QTYPE_ALIASES.keys()),
        help="quantization type used for SVD U/V tensors",
    )
    return parser.parse_args()


def copy_metadata(reader: GGUFReader, writer: GGUFWriter, file_type: LlamaFileType) -> None:
    for key, field in reader.fields.items():
        if key in ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]:
            continue
        if key == "general.file_type":
            writer.add_file_type(file_type)
            continue
        if key == "general.quantization_version":
            writer.add_quantization_version(field.parts[field.data[0]][0])
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


def is_mobile_svd_tensor(name: str) -> bool:
    return any(token in name for token in SVD_TENSOR_TOKENS)


def quantize_tensor_data(tensor, qtype: GGMLQuantizationType) -> np.ndarray:
    matrix = np.array(tensor.data, copy=False)
    matrix = np.ascontiguousarray(matrix.astype(np.float32, copy=False))
    return quantize(matrix, qtype)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    qtype = QTYPE_ALIASES[args.quant]
    output_path = Path(args.output) if args.output else input_path.with_suffix(input_path.suffix + f".mobile_{args.quant}.gguf")

    reader = GGUFReader(str(input_path))
    writer = GGUFWriter(str(output_path), "qwen2_svd")
    copy_metadata(reader, writer, FILE_TYPE_ALIASES[qtype])

    total_tensors = 0
    quantized_tensors = 0
    kept_tensors = 0
    input_bytes = 0
    output_bytes = 0

    for tensor in reader.tensors:
        total_tensors += 1
        input_bytes += tensor.n_bytes

        if is_mobile_svd_tensor(tensor.name):
            try:
                qdata = quantize_tensor_data(tensor, qtype)
            except Exception as exc:
                raise RuntimeError(f"failed to quantize tensor {tensor.name}: {exc}") from exc

            writer.add_tensor(
                name=tensor.name,
                tensor=qdata,
                raw_shape=tuple(int(x) for x in qdata.shape),
                raw_dtype=qtype,
            )
            quantized_tensors += 1
            output_bytes += qdata.nbytes
        else:
            raw = np.array(tensor.data, copy=False)
            writer.add_tensor(
                name=tensor.name,
                tensor=raw,
                raw_shape=tuple(int(x) for x in tensor.shape)[::-1],
                raw_dtype=tensor.tensor_type,
            )
            kept_tensors += 1
            output_bytes += raw.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"quant: {args.quant}")
    print(f"total_tensors: {total_tensors}")
    print(f"quantized_svd_tensors: {quantized_tensors}")
    print(f"kept_original_tensors: {kept_tensors}")
    print(f"input_tensor_bytes: {input_bytes}")
    print(f"output_tensor_bytes: {output_bytes}")
    print("FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
