from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
import re
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python strip_dense_ffn_from_svd.py <input_svd.gguf> <output_svd_compact.gguf>")
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    reader = GGUFReader(input_path)
    writer = GGUFWriter(output_path, "qwen2_svd")

    for key, field in reader.fields.items():
        if key in ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]:
            continue

        ftype = field.types[0]
        if str(ftype).endswith("STRING"):
            value = ''.join(chr(i) for i in field.parts[field.data[0]])
            writer.add_string(key=key, val=value)
        elif str(ftype).endswith("ARRAY"):
            writer.add_array(key=key, val=field.contents())
        else:
            value = field.parts[field.data[0]][0]
            writer.add_key_value(key=key, val=value, vtype=ftype)

    dense_ffn_pattern = re.compile(r"blk\.\d+\.ffn_(up|down|gate)\.weight$")

    kept = 0
    skipped = 0
    for tensor in reader.tensors:
        if dense_ffn_pattern.match(tensor.name):
            skipped += 1
            continue

        writer.add_tensor(
            name=tensor.name,
            tensor=tensor.data,
            raw_shape=tuple(tensor.shape)[::-1],
            raw_dtype=tensor.tensor_type,
        )
        kept += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"kept tensors: {kept}")
    print(f"skipped dense FFN tensors: {skipped}")
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
