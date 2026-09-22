# Standalone Llama transformer RTL core

`rtl/llama_decode_core.sv` is the first path in this repository where the
transformer forward pass is scheduled and calculated entirely in synthesizable
RTL. It does not call Python, DPI, `shortreal`, `$exp`, `$sqrt`, or vendor
floating-point simulation models during inference.

Implemented RTL operations:

- token embedding lookup and tied output projection;
- per-layer attention and feed-forward RMSNorm;
- Q, K, V and attention-output matrix-vector products;
- rotary position embedding (RoPE);
- persistent grouped-query KV cache;
- scaled causal Q/K scores, stable softmax and probability/V reduction;
- SwiGLU gate, up and down projections;
- both residual paths;
- all-layer sequencing, final RMSNorm, logits and argmax.

The arithmetic contract is signed Q16.16 by default. Multiplication rounds and
saturates. Exponential uses a deterministic range-limited repeated-squaring
approximation, division is signed fixed-point division, and square root is a
fixed-iteration integer square root. These are synthesizable implementations,
not IEEE FP32 and not bit-equivalent to the current NumPy reference.

## Run and synthesize the regression configuration

```bash
make -C asic llama-core-sim
make -C asic llama-core-synth
```

The regression instantiates a complete two-layer, two-head grouped-query
decoder with nonzero Q/K/V, attention output and SwiGLU matrices. It evaluates
two tokens, checks generated token IDs and checks persistent/resettable cache
position. This small configuration exists so the entire controller can run in
seconds. The synthesis target lowers the controller, memories and arithmetic
to a checked generic RTL netlist while deliberately preserving arithmetic
operators. Mapping the generic divider and nonlinear datapath all the way to
standard cells requires pipelined technology-specific implementations; blindly
expanding them combinationally produces an impractically large netlist.

## Parameter map

The configuration port writes 32-bit fixed-point words in this order:

1. RMS epsilon and attention scale;
2. tied token embedding/output matrix;
3. RoPE cosine and sine tables;
4. for every layer: attention norm, Q, K, V, O, FFN norm, gate, up, down;
5. final RMSNorm vector.

All matrices are row-major. `parameter_words` reports the instantiated word
count. Configuration writes are accepted only while idle. `clear_cache`
clears the RTL KV arrays and resets the token position.

## Honest boundary to Llama 3.2 1B

The controller graph is complete, but the checked-in regression does not yet
contain Meta's 1.235-billion trained parameters. Directly instantiating that
many 32-bit configuration words would require roughly 4.9 GB and would turn
them into standard-cell registers/muxes in the current open ASIC flow.

A model-specific physical implementation therefore still needs two things:

1. a converter/packer with an accuracy-qualified fixed-point or low-bit
   numerical format for every real tensor; and
2. physical ROM/SRAM macros connected to the parameter and KV interfaces,
   replacing the inferred arrays before full-size elaboration.

Python may prepare those immutable memory images and verify results, but it
must not perform RMSNorm, attention, MLP, KV updates, or layer scheduling in a
standalone inference run. The existing `llama_frozen_reference.py` remains a
golden/co-simulation program and is not part of this RTL core.
