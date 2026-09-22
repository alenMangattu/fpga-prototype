#!/usr/bin/env python3
"""Freeze an extracted Q8_0 Llama into an auditable, read-only ROM image.

This packages bytes; it does not design a physical mask ROM. The small generated
SV ROM is a synthesis proof using real model weights, not the complete decoder.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def freeze(source, output, tile_tensor, rows, first_row=0):
    original = json.loads((source / "manifest.json").read_text())
    meta = original["metadata"]
    if meta.get("general.architecture") != "llama":
        raise ValueError("expected a Llama model")
    tensors = original["tensors"]
    selected = next(t for t in tensors if t["name"] == tile_tensor)
    if (selected["type"] != "Q8_0" or len(selected["dimensions"]) != 2
            or rows < 1 or first_row < 0
            or first_row + rows > selected["dimensions"][1]):
        raise ValueError("tile must select valid rows of a Q8_0 matrix")
    # Validate all inputs before creating outputs.
    paths = []
    for index, tensor in enumerate(tensors):
        path = source / "tensors" / (
            f"{index:03d}_{tensor['name'].replace('/', '_')}.{tensor['type'].lower()}.bin")
        elements = math.prod(tensor["dimensions"])
        expected = {"F32": elements * 4, "Q8_0": elements // 32 * 34}.get(tensor["type"])
        if expected is None or expected != tensor["size_bytes"] or path.stat().st_size != expected:
            raise ValueError(f"invalid type/length: {path}")
        if tensor["type"] == "Q8_0" and tensor["dimensions"][0] % 32:
            raise ValueError(f"unaligned Q8_0 row: {path}")
        paths.append(path)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    with (output / "weights.rom.bin").open("wb") as dst:
        for tensor, path in zip(tensors, paths):
            dst.write(bytes((-dst.tell()) % 64))
            offset = dst.tell()
            sha = hashlib.sha256()
            with path.open("rb") as src:
                while chunk := src.read(4 * 1024 * 1024):
                    dst.write(chunk)
                    sha.update(chunk)
            entries.append({"name": tensor["name"], "dimensions": tensor["dimensions"],
                            "type": tensor["type"], "offset": offset,
                            "size_bytes": tensor["size_bytes"], "sha256": sha.hexdigest()})
    columns = selected["dimensions"][0]
    blocks = columns // 32
    selected_path = paths[tensors.index(selected)]
    with selected_path.open("rb") as f:
        f.seek(first_row * blocks * 34)
        tile = f.read(rows * blocks * 34)
    # Lowest 16 bits = FP16 scale; subsequent little-endian bytes are signed q[0:32].
    (output / "tile_weights.hex").write_text("".join(
        f"{int.from_bytes(tile[i:i+34], 'little'):068x}\n" for i in range(0, len(tile), 34)))
    # Literal case entries allow synthesis to absorb the fixed values into logic.
    addr_bits = max(1, (rows * blocks - 1).bit_length())
    cases = "".join(f"            {addr_bits}'d{i // 34}: data = 272'h{int.from_bytes(tile[i:i+34], 'little'):068x};\n"
                    for i in range(0, len(tile), 34))
    (output / "llama_tile_rom.sv").write_text(
        "// Generated from real Llama weights. No programming/write interface.\n"
        f"module llama_tile_rom(input wire [{addr_bits-1}:0] address, output reg [271:0] data);\n"
        "    always @* begin\n        case (address)\n" + cases +
        "            default: data = 272'd0;\n        endcase\n    end\nendmodule\n")
    (output / "tile_config.svh").write_text(
        f"`define LLAMA_TILE_COLUMNS {columns}\n`define LLAMA_TILE_ROWS {rows}\n"
        f"`define LLAMA_TILE_SHA256 256'h{hashlib.sha256(tile).hexdigest()}\n")
    config = {k: v for k, v in meta.items() if not k.startswith("tokenizer.")}
    manifest = {"format": "llama-frozen-rom-v1", "source": str(source.resolve()),
                "source_manifest_sha256": digest(source / "manifest.json"),
                "parameter_count": sum(math.prod(t["dimensions"]) for t in tensors),
                "rom_bytes": (output / "weights.rom.bin").stat().st_size,
                "rom_sha256": digest(output / "weights.rom.bin"),
                "metadata": config, "tensors": entries,
                "tile": {"tensor": tile_tensor, "first_row": first_row, "rows": rows,
                         "columns": columns, "blocks_per_row": blocks,
                         "weight_bits": len(tile) * 8, "sha256": hashlib.sha256(tile).hexdigest()},
                "hardware_scope": "Selected matrix tile only; full decoder is a software reference."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "tokenizer.json").write_text(json.dumps(
        {k: v for k, v in meta.items() if k.startswith("tokenizer.")}) + "\n")
    for name in ("LICENSE-LLAMA-3.2.txt", "ACCEPTABLE-USE-POLICY.txt", "NOTICE.txt"):
        (output / name).write_bytes((source / name).read_bytes())
    print(json.dumps({k: manifest[k] for k in ("parameter_count", "rom_bytes", "rom_sha256", "tile")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("models/llama3.2-1b-int8"))
    p.add_argument("--output", type=Path, default=Path("asic/build/llama_frozen"))
    p.add_argument("--tile-tensor", default="blk.0.attn_q.weight")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--first-row", type=int, default=0)
    a = p.parse_args()
    if a.output.resolve() == a.source.resolve() or a.source.resolve() in a.output.resolve().parents:
        p.error("output must be outside the source model directory")
    freeze(a.source, a.output, a.tile_tensor, a.rows, a.first_row)
