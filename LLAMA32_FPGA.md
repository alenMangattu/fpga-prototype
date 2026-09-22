# Llama 3.2 1B FPGA target

The fixed-model ASIC proof now has a separate [ROM and co-simulation flow](asic/LLAMA_FROZEN.md).
It runs the full model in a CPU reference and uses a synthesized, fixed-weight
matrix tile in RTL co-simulation; it is not a complete standalone decoder.

The copied model is the Ollama `llama3.2:1b` Q8_0 GGUF. Its SHA-256 is
`74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45`.

## Model graph

- 16 decoder blocks
- model width 2048
- 32 query heads and 8 key/value heads (GQA)
- head width 64
- RoPE position encoding, base 500000
- RMSNorm with epsilon 1e-5
- gated SwiGLU MLP, width 8192
- vocabulary 128256

This is not the GPT-2 graph. A compatible decoder block is:

1. `u = RMSNorm(x)`
2. `q = RoPE(Wq*u)`, `k = RoPE(Wk*u)`, `v = Wv*u`
3. `a = causal_softmax(q*k^T/sqrt(64))*v` using grouped-query attention
4. `r = x + Wo*a`
5. `n = RMSNorm(r)`
6. `y = r + Wdown*(SiLU(Wgate*n) * (Wup*n))`

## Memory architecture

The GGUF is 1,321,082,688 bytes. It cannot reside in the Ti60's on-chip RAM.
Weights must remain in external DRAM and stream through a tiled Q8_0 matrix
engine. On-chip RAM stores activation tiles, quantization blocks, and the KV
cache working set. The full 131072-token advertised context is not practical
on a small FPGA; the deployed context must be capped according to external
memory capacity and bandwidth.

`rtl/q8_0_block_mac.sv` is the first native weight datapath. It consumes the
32 signed bytes in one GGML Q8_0 block, computes the exact integer dot product,
and forwards the FP16 weight and activation scales. The next pipeline stage
converts/scales and accumulates block results.

## Prepare the model

```bash
python3 tools/gguf_split.py \
  models/llama3.2-1b/model-Q8_0.gguf \
  --manifest models/llama3.2-1b/manifest.json \
  --extract-dir models/llama3.2-1b/tensors
```

The tensor files stay in native GGML encoding; they are not giant Verilog ROM
initializers. A board host or DMA engine loads them using the offsets, shapes,
and types in `manifest.json`.

The copied artifact can also be checked independently through Ollama:

```bash
ollama create llama3.2-fpga-source -f models/llama3.2-1b/Modelfile
ollama run llama3.2-fpga-source
```

## Test the FPGA Q8_0 datapath

```bash
iverilog -g2012 -Wall -s tb_q8_0_block_fp32 \
  -o q8_fp32_test \
  rtl/q8_0_block_mac.sv rtl/fp16_to_fp32.sv \
  rtl/int32_to_fp32.sv rtl/fp32_mul.sv \
  rtl/q8_0_block_fp32.sv tb/tb_q8_0_block_fp32.sv
vvp q8_fp32_test
```

## Current boundary

The model is locally packaged and the Q8_0 compute primitive is synthesizable.
A complete standalone token generator still requires the board-specific DRAM
controller/DMA, tokenizer host, sampling code, RMSNorm/RoPE/softmax/SiLU
pipelines, KV-cache controller, and integration with generated FPGA IP. Those
interfaces depend on the exact board and its external-memory hardware.
