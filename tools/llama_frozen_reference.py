#!/usr/bin/env python3
"""CPU numerical reference for the frozen Llama ROM, NOT an RTL decoder.

Uses all 16 trained layers, original Q8_0 scales, RMSNorm, GGUF interleaved
RoPE and its stored frequency factors, GQA, causal softmax, SwiGLU, tied output
weights, and an FP32 KV cache. Dense FP32 weights are cached (~5 GB for 1B).
"""
import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import regex

Q8 = np.dtype([("scale", "<f2"), ("q", "i1", (32,))])


class Tokenizer:
    def __init__(self, root):
        m = json.loads((root / "tokenizer.json").read_text())
        if m["tokenizer.ggml.pre"] != "llama-bpe":
            raise ValueError("only llama-bpe is supported")
        self.tokens = m["tokenizer.ggml.tokens"]
        self.ids = {s: i for i, s in enumerate(self.tokens)}
        self.ranks = {tuple(s.split(" ")): i for i, s in enumerate(m["tokenizer.ggml.merges"])}
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = list(bs)
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + len(cs) - 188)
        self.encode_byte = {b: chr(c) for b, c in zip(bs, cs)}
        self.decode_byte = {v: k for k, v in self.encode_byte.items()}
        self.bos = m["tokenizer.ggml.bos_token_id"]
        self.stops = {m["tokenizer.ggml.eos_token_id"], self.ids["<|end_of_text|>"], self.ids["<|eot_id|>"]}
        self.pattern = regex.compile(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

    def encode(self, text):
        result = []
        for word in self.pattern.findall(text):
            parts = [self.encode_byte[b] for b in word.encode("utf-8")]
            while len(parts) > 1:
                best = min(range(len(parts)-1), key=lambda i: self.ranks.get((parts[i], parts[i+1]), math.inf))
                if (parts[best], parts[best+1]) not in self.ranks:
                    break
                parts[best:best+2] = [parts[best] + parts[best+1]]
            result.extend(self.ids[s] for s in parts)
        return result

    def prompt(self, text, chat):
        if not chat:
            return [self.bos] + self.encode(text)
        return ([self.bos, self.ids["<|start_header_id|>"]] + self.encode("user")
                + [self.ids["<|end_header_id|>"]] + self.encode("\n\n" + text)
                + [self.ids["<|eot_id|>"], self.ids["<|start_header_id|>"]]
                + self.encode("assistant") + [self.ids["<|end_header_id|>"]] + self.encode("\n\n"))

    def decode(self, ids):
        chunks = bytearray()
        for i in ids:
            if self.tokens[i].startswith("<|"):
                chunks.extend(self.tokens[i].encode())
            else:
                chunks.extend(self.decode_byte[c] for c in self.tokens[i])
        return chunks.decode("utf-8", errors="replace")


class FrozenLlama:
    def __init__(self, root, context=128, rtl_tile=None):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest["format"] != "llama-frozen-rom-v1":
            raise ValueError("unsupported ROM format")
        m = self.manifest["metadata"]
        self.d = m["llama.embedding_length"]
        self.layers = m["llama.block_count"]
        self.heads = m["llama.attention.head_count"]
        self.kv_heads = m["llama.attention.head_count_kv"]
        self.hd = self.d // self.heads
        self.eps = np.float32(m["llama.attention.layer_norm_rms_epsilon"])
        if context < 1 or context > m["llama.context_length"]:
            raise ValueError("invalid context size")
        self.rom = np.memmap(root / "weights.rom.bin", dtype="u1", mode="r")
        if len(self.rom) != self.manifest["rom_bytes"]:
            raise ValueError("ROM length does not match manifest")
        self.tensors = {t["name"]: t for t in self.manifest["tensors"]}
        self.cache = {}
        self.rtl_tile = rtl_tile.resolve() if rtl_tile else None
        self.rtl_invocations = 0
        self.rtl_cycles = 0
        self.keys = np.zeros((self.layers, context, self.kv_heads, self.hd), dtype=np.float32)
        self.values = np.zeros_like(self.keys)
        self.position = 0
        factors = self.weight("rope_freqs.weight") if "rope_freqs.weight" in self.tensors else np.ones(self.hd//2)
        self.freq = (np.float32(m["llama.rope.freq_base"]) **
                     (-np.arange(0, self.hd, 2, dtype=np.float32) / self.hd)) / factors

    def blocks(self, name):
        t = self.tensors[name]
        if t["type"] != "Q8_0":
            raise ValueError("expected Q8_0")
        return np.ndarray((t["size_bytes"] // 34,), dtype=Q8, buffer=self.rom, offset=t["offset"])

    def weight(self, name):
        if name not in self.cache:
            t = self.tensors[name]
            if t["type"] == "F32":
                w = np.ndarray(tuple(reversed(t["dimensions"])), dtype="<f4", buffer=self.rom, offset=t["offset"])
            else:
                b = self.blocks(name)
                w = b["q"].astype(np.float32)
                w *= b["scale"].astype(np.float32)[:, None]
                w = w.reshape(tuple(reversed(t["dimensions"])))
            self.cache[name] = w
        return self.cache[name]

    def embedding(self, token):
        if not 0 <= token < self.tensors["token_embd.weight"]["dimensions"][1]:
            raise ValueError("token outside vocabulary")
        # Avoid expanding the tied output matrix just to fetch an embedding.
        b = self.blocks("token_embd.weight")[token*self.d//32:(token+1)*self.d//32]
        return (b["q"].astype(np.float32) * b["scale"].astype(np.float32)[:, None]).ravel()

    def norm(self, x, name):
        return x * np.float32(1 / np.sqrt(np.mean(x*x, dtype=np.float32) + self.eps)) * self.weight(name)

    def rope(self, x):
        angle = np.float32(self.position) * self.freq
        c, s = np.cos(angle), np.sin(angle)
        even, odd = x[:, 0::2].copy(), x[:, 1::2].copy()
        x[:, 0::2] = even*c - odd*s
        x[:, 1::2] = even*s + odd*c
        return x

    def rtl_projection(self, x):
        """Use outputs from the compiled fixed-ROM Verilog, with Q8 activations.

        Only the selected matrix rows run in RTL; the rest of the network is CPU.
        Each invocation gets isolated files and verifies the compiled ROM hash.
        """
        tile = self.manifest["tile"]
        xb = x.reshape(-1, 32)
        scales = (np.max(np.abs(xb), axis=1) / 127).astype(np.float16)
        scales[scales == 0] = np.float16(1)
        q = np.clip(np.rint(xb / scales.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
        words = [int.from_bytes(scales[i:i+1].astype("<f2").tobytes() + q[i].tobytes(), "little")
                 for i in range(len(scales))]
        with tempfile.TemporaryDirectory(prefix="llama-rtl-") as work:
            path = Path(work)
            (path / "activations.hex").write_text("".join(f"{v:068x}\n" for v in words))
            proc = subprocess.run(["vvp", str(self.rtl_tile)], cwd=path, text=True,
                                  capture_output=True, timeout=120, check=True)
            marker = f"ROM_SHA256={tile['sha256']} CYCLES="
            line = next((s for s in proc.stdout.splitlines() if s.startswith(marker)), None)
            if line is None:
                raise ValueError("compiled RTL ROM does not match this frozen model")
            values = [int(s, 16) for s in (path / "result.hex").read_text().splitlines()]
            if len(values) != tile["rows"]:
                raise ValueError("RTL returned the wrong number of rows")
            result = np.array(values, dtype=np.uint32).view(np.float32)
            if not np.isfinite(result).all():
                raise ValueError("RTL returned non-finite results")
            self.rtl_invocations += 1
            self.rtl_cycles += int(line.split("CYCLES=")[1])
            return result

    def linear(self, name, x):
        y = self.weight(name) @ x
        if self.rtl_tile and self.manifest["tile"]["tensor"] == name:
            tile = self.manifest["tile"]
            lo = tile["first_row"]
            y[lo:lo+tile["rows"]] = self.rtl_projection(x)
        return y

    def step(self, token, logits=True):
        p = self.position
        if p >= self.keys.shape[1]:
            raise ValueError("KV context exhausted")
        x = self.embedding(token)
        for layer in range(self.layers):
            prefix = f"blk.{layer}."
            linear = lambda name, v: self.linear(prefix + name + ".weight", v)
            n = self.norm(x, prefix + "attn_norm.weight")
            query = linear("attn_q", n)
            q = self.rope(query.reshape(self.heads, self.hd))
            k = self.rope(linear("attn_k", n).reshape(self.kv_heads, self.hd))
            v = linear("attn_v", n).reshape(self.kv_heads, self.hd)
            self.keys[layer, p] = k
            self.values[layer, p] = v
            groups = self.heads // self.kv_heads
            q = q.reshape(self.kv_heads, groups, self.hd)
            scores = np.einsum("hgd,thd->hgt", q, self.keys[layer, :p+1]) / np.float32(np.sqrt(self.hd))
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            attn = np.einsum("hgt,thd->hgd", probs, self.values[layer, :p+1]).reshape(self.d)
            x = x + linear("attn_output", attn)
            n = self.norm(x, prefix + "ffn_norm.weight")
            gate, up = linear("ffn_gate", n), linear("ffn_up", n)
            # Stable SiLU without exponent overflow.
            silu = gate * np.exp(-np.logaddexp(np.float32(0), -gate))
            x = x + linear("ffn_down", silu * up)
        self.position += 1
        if not logits:
            return None
        name = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        out = self.linear(name, self.norm(x, "output_norm.weight"))
        if not np.isfinite(out).all():
            raise ValueError("non-finite logits")
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("asic/build/llama_frozen"))
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--report", type=Path)
    p.add_argument("--rtl-tile", type=Path, help="compiled tb_llama_frozen_infer vvp; enables hybrid CPU/RTL execution")
    p.add_argument("--rtl-clock-mhz", type=float, default=50.0,
                   help="hypothetical RTL clock used to convert simulated cycles to time (default: 50)")
    a = p.parse_args()
    if not math.isfinite(a.rtl_clock_mhz) or a.rtl_clock_mhz <= 0:
        p.error("--rtl-clock-mhz must be a positive finite number")
    tok = Tokenizer(a.model)
    ids = tok.prompt(a.prompt, a.chat)
    if a.tokens < 1 or len(ids) + a.tokens > a.context:
        p.error("prompt + output must fit context; tokens must be positive")
    model = FrozenLlama(a.model, a.context, a.rtl_tile)
    started = time.monotonic()
    backend = "hybrid CPU + fixed-weight RTL tile" if a.rtl_tile else "CPU NumPy reference; not RTL inference"
    print(f"{backend}: {model.layers} layers; {len(ids)} prompt tokens; ROM {model.manifest['rom_sha256']}", flush=True)
    for i, token in enumerate(ids):
        logits = model.step(token, logits=(i == len(ids)-1))
    output = []
    top = np.argsort(logits)[-5:][::-1]
    first_top = [{"id": int(i), "text": tok.decode([i]), "logit": float(logits[i])} for i in top]
    for i in range(a.tokens):
        token = int(np.argmax(logits))
        if token in tok.stops:
            break
        output.append(token)
        print(tok.decode([token]), end="", flush=True)
        if i + 1 < a.tokens:
            logits = model.step(token)
    print()
    rtl_estimated_seconds = (model.rtl_cycles / (a.rtl_clock_mhz * 1_000_000)
                             if a.rtl_tile else None)
    report = {"backend": backend, "rom_sha256": model.manifest["rom_sha256"],
              "prompt": a.prompt, "chat": a.chat, "input_ids": ids, "output_ids": output,
              "output_text": tok.decode(output), "first_top5": first_top,
              "seconds": time.monotonic() - started,
              "rtl_invocations": model.rtl_invocations, "rtl_compute_cycles": model.rtl_cycles,
              "rtl_assumed_clock_mhz": a.rtl_clock_mhz if a.rtl_tile else None,
              "rtl_estimated_seconds": rtl_estimated_seconds,
              "rtl_scope": model.manifest["tile"] if a.rtl_tile else None}
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(output)} generated tokens in {report['seconds']:.2f}s (host CPU time).")
    if a.rtl_tile:
        print(f"RTL tile: {model.rtl_cycles:,} simulated cycles across "
              f"{model.rtl_invocations} invocations; {rtl_estimated_seconds * 1e6:,.3f} us "
              f"at an assumed {a.rtl_clock_mhz:g} MHz.")
        print("Hardware estimate covers only the fixed 8-row matrix tile; full-model hardware time is unavailable.")


if __name__ == "__main__":
    main()
