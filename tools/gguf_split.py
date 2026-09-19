#!/usr/bin/env python3
"""Inspect a GGUF file and optionally split its tensor payloads.

The extractor deliberately preserves tensors in their native GGML encoding.
For this project that means Q8_0 weights remain 34-byte blocks containing one
FP16 scale followed by 32 signed bytes, ready for streaming to the FPGA.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


VALUE_TYPES = {
    0: ("uint8", "<B"),
    1: ("int8", "<b"),
    2: ("uint16", "<H"),
    3: ("int16", "<h"),
    4: ("uint32", "<I"),
    5: ("int32", "<i"),
    6: ("float32", "<f"),
    7: ("bool", "<?"),
    10: ("uint64", "<Q"),
    11: ("int64", "<q"),
    12: ("float64", "<d"),
}

GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
}


def read_exact(handle, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise EOFError(f"wanted {size} bytes, received {len(value)}")
    return value


def read_u32(handle) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_string(handle) -> str:
    return read_exact(handle, read_u64(handle)).decode("utf-8")


def read_value(handle, value_type: int):
    if value_type in VALUE_TYPES:
        name, fmt = VALUE_TYPES[value_type]
        return name, struct.unpack(fmt, read_exact(handle, struct.calcsize(fmt)))[0]
    if value_type == 8:
        return "string", read_string(handle)
    if value_type == 9:
        element_type = read_u32(handle)
        count = read_u64(handle)
        values = [read_value(handle, element_type)[1] for _ in range(count)]
        return "array", values
    raise ValueError(f"unsupported GGUF metadata value type {value_type}")


def tensor_size(dimensions: list[int], ggml_type: int) -> int:
    if ggml_type not in GGML_TYPES:
        raise ValueError(f"unsupported GGML tensor type {ggml_type}")
    elements = 1
    for dimension in dimensions:
        elements *= dimension
    _, block_elements, block_bytes = GGML_TYPES[ggml_type]
    if elements % block_elements:
        raise ValueError(f"{elements} elements do not fill a quantization block")
    return elements // block_elements * block_bytes


def sanitize(name: str) -> str:
    return name.replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()

    with args.gguf.open("rb") as source:
        if read_exact(source, 4) != b"GGUF":
            raise ValueError("input is not a GGUF file")
        version = read_u32(source)
        tensor_count = read_u64(source)
        metadata_count = read_u64(source)

        metadata = {}
        metadata_types = {}
        for _ in range(metadata_count):
            key = read_string(source)
            value_type = read_u32(source)
            type_name, value = read_value(source, value_type)
            metadata[key] = value
            metadata_types[key] = type_name

        tensors = []
        for _ in range(tensor_count):
            name = read_string(source)
            dimension_count = read_u32(source)
            dimensions = [read_u64(source) for _ in range(dimension_count)]
            ggml_type = read_u32(source)
            relative_offset = read_u64(source)
            type_name = GGML_TYPES.get(ggml_type, (f"TYPE_{ggml_type}", 0, 0))[0]
            tensors.append({
                "name": name,
                "dimensions": dimensions,
                "ggml_type": ggml_type,
                "type": type_name,
                "relative_offset": relative_offset,
                "size_bytes": tensor_size(dimensions, ggml_type),
            })

        alignment = int(metadata.get("general.alignment", 32))
        data_offset = (source.tell() + alignment - 1) // alignment * alignment
        for tensor in tensors:
            tensor["file_offset"] = data_offset + tensor["relative_offset"]

        parameter_count = metadata.get("general.parameter_count")
        if parameter_count is None:
            parameter_count = sum(
                __import__("math").prod(tensor["dimensions"])
                for tensor in tensors
            )

        manifest = {
            "source": str(args.gguf),
            "gguf_version": version,
            "alignment": alignment,
            "tensor_data_offset": data_offset,
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "metadata": metadata,
            "metadata_types": metadata_types,
            "tensors": tensors,
        }

        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

        if args.extract_dir:
            args.extract_dir.mkdir(parents=True, exist_ok=True)
            for index, tensor in enumerate(tensors):
                source.seek(tensor["file_offset"])
                remaining = tensor["size_bytes"]
                target_path = args.extract_dir / f"{index:03d}_{sanitize(tensor['name'])}.{tensor['type'].lower()}.bin"
                with target_path.open("wb") as target:
                    while remaining:
                        block = read_exact(source, min(remaining, args.chunk_size))
                        target.write(block)
                        remaining -= len(block)

    summary = {
        "architecture": metadata.get("general.architecture"),
        "parameters": parameter_count,
        "tensor_count": tensor_count,
        "gguf_version": version,
        "model_bytes": args.gguf.stat().st_size,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
