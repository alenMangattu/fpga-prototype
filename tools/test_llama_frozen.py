#!/usr/bin/env python3
"""Verify every frozen tensor; prepare independent NumPy RTL test vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from llama_frozen_reference import FrozenLlama, Tokenizer


def write_hex(path, data, width):
    path.write_text("".join(f"{int(v):0{width}x}\n" for v in data))


def prepare(root):
    m = json.loads((root / "manifest.json").read_text())
    with (root / "weights.rom.bin").open("rb") as f:
        actual_hash = hashlib.file_digest(f, "sha256").hexdigest()
        if actual_hash != m["rom_sha256"]:
            raise AssertionError("ROM checksum mismatch")
        for t in m["tensors"]:
            f.seek(t["offset"])
            if hashlib.sha256(f.read(t["size_bytes"])).hexdigest() != t["sha256"]:
                raise AssertionError(f"tensor checksum mismatch: {t['name']}")
    model = FrozenLlama(root, context=4)
    tile = m["tile"]
    rows, blocks = tile["rows"], tile["blocks_per_row"]
    b = model.blocks(tile["tensor"]).reshape(-1, blocks)[tile["first_row"]:tile["first_row"]+rows]
    # Ensure the literal ROM/hex representation is identical to the frozen bytes.
    literal = bytes().join(int(line, 16).to_bytes(34, "little") for line in (root / "tile_weights.hex").read_text().splitlines())
    assert literal == b.tobytes()
    assert hashlib.sha256(literal).hexdigest() == tile["sha256"]
    tok = Tokenizer(root)
    assert len(tok.encode_byte) == len(tok.decode_byte) == 256
    for text in ("The capital of France is", "Hello, world!", "àé 日本語 123456\n\n", " a  b\t\n"):
        assert tok.decode(tok.encode(text)) == text
    vectors = []
    if tile["tensor"] == "blk.0.attn_q.weight" and tile["columns"] == model.d:
        for token in (tok.bos, tok.encode("The")[0]):
            x = model.norm(model.embedding(token), "blk.0.attn_norm.weight").reshape(blocks, 32)
            scales = (np.max(np.abs(x), axis=1) / 127).astype(np.float16)
            scales[scales == 0] = np.float16(1)
            q = np.clip(np.rint(x / scales.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
            vectors.append((f"normalized_embedding_token_{token}", q, scales))
    rng = np.random.default_rng(173)
    vectors.extend([
        ("zero", np.zeros((blocks, 32), dtype=np.int8), np.ones(blocks, dtype=np.float16)),
        ("positive_limit", np.full((blocks, 32), 127, dtype=np.int8), np.full(blocks, .125, dtype=np.float16)),
        ("negative_limit", np.full((blocks, 32), -128, dtype=np.int8), np.full(blocks, .125, dtype=np.float16)),
        ("random_signed", rng.integers(-128, 128, size=(blocks, 32), dtype=np.int8),
         rng.uniform(.0001, .03, blocks).astype(np.float16)),
        ("fp16_subnormal_scale", rng.integers(-128, 128, size=(blocks, 32), dtype=np.int8),
         np.full(blocks, np.float16(2**-24), dtype=np.float16)),
    ])
    activation_words, expected, diagnostics = [], [], []
    for name, q, scales in vectors:
        for i in range(blocks):
            word = scales[i:i+1].astype("<f2").tobytes() + q[i].tobytes()
            activation_words.append(int.from_bytes(word, "little"))
        # Independent exact signed integer dot and explicitly rounded FP32 schedule.
        dot = (b["q"].astype(np.int32) * q.astype(np.int32)[None, :, :]).sum(axis=2, dtype=np.int32)
        values = (dot.astype(np.float32) * b["scale"].astype(np.float32)) * scales.astype(np.float32)
        totals = np.zeros(rows, dtype=np.float32)
        for i in range(blocks):
            totals = totals + values[:, i]
        expected.extend(totals.view(np.uint32))
        diagnostics.append({"input": name, "expected_fp32": [float(v) for v in totals]})
    write_hex(root / "activations.hex", activation_words, 68)
    write_hex(root / "expected.hex", expected, 8)
    (root / "test_config.svh").write_text(f"`define LLAMA_TEST_CASES {len(vectors)}\n")
    report = {"rom_sha256": actual_hash, "verified_tensors": len(m["tensors"]),
              "rtl_test_vectors": len(vectors), "rows_per_vector": rows,
              "comparisons": len(expected), "vectors": diagnostics}
    (root / "vectors_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"ROM verified: {len(m['tensors'])} tensors. Prepared {len(vectors)} vectors, {len(expected)} row comparisons.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("asic/build/llama_frozen"))
    a = p.parse_args()
    prepare(a.model)
