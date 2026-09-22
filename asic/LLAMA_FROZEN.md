# Fixed-weight Llama proof

For the newer standalone transformer execution controller, see
`asic/LLAMA_RTL_CORE.md` and `rtl/llama_decode_core.sv`. Unlike the hybrid flow
below, that core schedules the complete transformer graph in RTL; its current
regression uses a small Q16.16 model while full-size Q8 memory integration and
accuracy qualification remain separate work.

This flow uses the actual local Llama 3.2 1B Instruct Q8_0 weights. It creates
a complete frozen weight image, runs the complete model on the CPU, and
executes a selected matrix tile in synthesizable, fixed-weight Verilog. The
co-simulation feeds the RTL results back into the model's forward pass.

**It is not yet a complete Llama ASIC, a fabricated chip, or a reproduction
of Taalas's implementation.** The whole decoder has not been synthesized.
No physical area, clock frequency, power or token/s claim follows from this
generic synthesis run. Existing SKY130 metrics belong to the separate tiny
transformer, not this Llama design.

## Run

From the project root, using the existing virtual environment:

```bash
.venv/bin/python -m pip install -r tools/requirements-llama.txt
make -C asic llama-sim          # fixed ROM RTL + independent numerical checks
make -C asic llama-cosim        # full model, with selected rows executed in RTL
make -C asic llama-netlist-sim  # synthesis + the same tests on the gate netlist
make -C asic llama-layout-preview # placed/clocked computer layout + SVG views
make -C asic llama-gds          # attempt routed SKY130 GDS for the 1-row tile
```

`make -C asic llama-run` runs just the full CPU reference. The reference caches
approximately 5 GB of dequantized FP32 weights in RAM. This is a reference
implementation, not a benchmark of optimized inference software.

For another prompt after building the co-simulation:

```bash
.venv/bin/python tools/llama_frozen_reference.py \
  --chat --prompt "What is 2 + 2?" --tokens 16 \
  --rtl-tile asic/build/llama_frozen/infer_tile \
  --rtl-clock-mhz 50
```

The program reports host execution time and a separate cycle-derived RTL tile
time. `--rtl-clock-mhz` defaults to 50 MHz and is an assumption, since this
Llama tile has generic synthesis results but no placed-and-routed timing result.
The estimate covers only the selected eight matrix rows. A whole-model hardware
latency cannot be reported until the complete decoder datapath is implemented.

The default context cap is 128 tokens including the prompt and output. The
simple chat mode wraps one user message and an assistant header; it does not
implement tools or the full date-dependent Ollama system template. The default
plain completion uses a BOS token and the supplied text. Both use greedy decoding.

## What is fixed in hardware

The default tile contains rows 0–7 of `blk.0.attn_q.weight`: **8 × 2048 trained
weights**, plus their original 512 FP16 block scales (139,264 bits total).
The generated `llama_tile_rom.sv` expresses those bytes as literal constants.
There is no weight-write port and no run-time weight-loading operation in the
tile. Generic synthesis reduces the constant ROM to gates/multiplexers.
This is different from a foundry mask-ROM macro; integrating such a macro
requires a technology-specific memory implementation.

The activation interface writes 64 blocks, each containing an FP16 scale and
32 signed INT8 activations. A start is accepted only when every activation
block is initialized and the tile is idle. Writes and further starts during
computation are ignored. Results have one-cycle valid pulses with no
backpressure, and reset clears validity and aborts an in-flight operation.

Each cycle issues 32 signed multiplies. A six-stage balanced tree produces
the exact integer dot. The existing FP32 units apply the two FP16 scales
and accumulate 64 blocks per output row. The tested tile takes **519 compute
cycles for eight rows**, excluding activation loading and the start edge.
The FP32 scale/accumulate logic still needs timing-driven implementation work;
this cycle count does not establish an achievable clock rate.

```text
Literal trained weight ROM ─┐
                           ├─ 32-lane pipelined dot ─ scales ─ FP32 row sum
Writable activation buffer ┘                                  │
                                                              v
                   CPU Llama forward pass consumes these rows in co-simulation
```

The tile can be rebuilt for other rows or another Q8_0 matrix. For example:

```bash
make -C asic llama-sim llama-cosim \
  LLAMA_TENSOR=blk.1.ffn_down.weight LLAMA_FIRST_ROW=16 LLAMA_ROWS=8
```

This replaces the generated tile in `asic/build/llama_frozen`; rebuild with
default arguments to return to the default query tile. Enlarging `LLAMA_ROWS`
increases the literal ROM and synthesis cost; it does not replicate compute
lanes or implement a full decoder. The co-simulator checks the compiled ROM's
hash before accepting its results.

## Full model packaging and execution

`tools/freeze_llama.py` preserves every Q8_0 and F32 tensor, gives it an aligned
offset, records a SHA-256 for every tensor and the entire image, and retains
the model notices/license. The default frozen image contains:

- 147 tensors and 1,235,814,432 stored parameter elements.
- 1,313,251,456 bytes including alignment, roughly 10.51 billion bits.
- Q8_0 blocks: little-endian FP16 scale followed by 32 signed INT8 values.
- F32 norm and RoPE tensors, without lossy conversion.

The full CPU reference reads that frozen image, not Ollama. It implements all
16 decoder blocks, RMSNorm, RoPE with the supplied frequency factors,
grouped-query causal softmax attention, the KV cache, SwiGLU and tied embedding/
output weights. Its graph and GGUF conventions can be compared with
[llama.cpp's Llama implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/llama.cpp)
and [Q8_0 implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c).

The reference dequantizes Q8_0 weights and uses FP32 activations. The hardware
tile additionally quantizes activations per 32 values to INT8 with FP16 scales.
Therefore hybrid inference is numerically different from the pure CPU
reference. A matching text smoke test is not an accuracy evaluation; deploying
W8A8 throughout the decoder requires wider numerical and task-quality testing.

## Verification and artifacts

All generated files are in `asic/build/llama_frozen/` (ignored by git):

| File | Purpose |
|---|---|
| `weights.rom.bin`, `manifest.json` | All frozen weights, shapes, offsets and hashes |
| `llama_tile_rom.sv`, `tile_config.svh` | Literal selected weights and dimensions |
| `vectors_report.json` | Verified image and independent test-vector expectations |
| `reference_run.json`, `cosim_run.json` | Prompt, token IDs, backend and execution scope |
| `tile_netlist.v`, `tile_netlist.json`, `synthesis.log` | Generic synthesized tile |

The arithmetic regression checks 213 signed block dots with extrema, bubbles,
scale forwarding and tags. The default tile regression checks 64 bit-exact
FP32 row outputs (seven input vectors plus a replay), busy protection,
uninitialized-input rejection and reset during execution. Inputs include
actual normalized embeddings, random signed values and subnormal FP16 scales.
The expected FP32 results are independently calculated with NumPy using the
same specified sequence of integer dots and FP32 roundings.

The default tile passed both the RTL regression and the same 64 comparisons
on the synthesized gate netlist. Yosys `check -assert` reported zero problems.
The netlist contains 100,971 generic logic/register cells (excluding eight
scope-information records), with no remaining memory objects or black boxes.
These are generic gates, not a SKY130 standard-cell area or timing result.

The initial CPU and hybrid smoke runs both completed:

```text
Prompt: The capital of France is
Output:  Paris. The Eiffel Tower is
```

The installed Ollama model using the same source GGUF also produced these
eight tokens with raw prompting and temperature zero. The hybrid run invoked
RTL 13 times: six prompt-token evaluations and seven subsequent decode
evaluations. Its total 6,747 RTL cycles account only for the selected tile;
all other matrix rows and all other operators execute on the CPU.

## What remains before a full fixed-model chip

The remaining work is substantial: distribute the entire 10.51-Gbit weight
image across a physical memory/compute architecture; implement all remaining
matrix computations, RMSNorm, RoPE, softmax, SiLU, KV storage and scheduling in
RTL; verify full-decoder numerical quality; then map to a selected PDK and
memory macros and complete physical design, timing and signoff. Tokenization
and sampling can remain on a host, but the decoder computation must not depend
on the CPU reference for a standalone inference chip.

A foundry/process and usable ROM/SRAM macros must be selected before the full
model's physical implementation can be determined. Setting a giant Verilog
array to these weights would not establish that it fits or routes on a chip.

## From this repository to manufactured silicon

`make -C asic llama-gds` runs one trained 2,048-weight output row through the
open SKY130 PDK at a conservative initial 10 MHz constraint and a 1.5 mm square
exploratory die. It uses `asic/build/llama_gds_frozen`, leaving the default
eight-row co-simulation artifacts untouched. Set `LLAMA_GDS_ROWS` to explore a
larger physical tile; the eight-row standard-cell ROM has severe routing
congestion and is not the default physical proof.
The preview result is a computer prototype of the floorplan, placed standard
cells, power grid and clock tree. Inspect reports under `asic/sky130/runs/` and
the generated SVGs under the newest run's `render/` directory. A generated GDS
is not tapeout-ready unless all required timing, DRC, LVS, antenna and power
checks pass.

The measured one-row standard-cell experiment reached synthesis, placement,
clock-tree synthesis and global routing. It had about 0.974 mm² of functional
cell area before clock/hold repair, no setup violations at the 10 MHz target,
and about 19,000 initial hold endpoints. Hold repair added roughly 36% area.
Detailed routing remained heavily congested: its first pass reported 51,006
violations and the first optimization pass reduced this to 19,974. The run was
stopped because further passes were taking hours and did not offer a practical
path to a clean layout. This is a useful physical prototype, not a clean GDS.

To fabricate it, the design also needs a pad ring, ESD structures, power pads,
clock/reset I/O, a package and a test plan. Then choose either an MPW shuttle
for a few prototypes or a dedicated wafer run, follow that service's harness
and submission rules, run foundry signoff tools, and submit the final GDS/OASIS
plus its required reports. The exact pad cells, seal ring and signoff deck are
process- and shuttle-specific, so they cannot be finalized before choosing the
fabrication service.

For the complete 1B model, the 10.51-Gbit frozen image dominates the chip.
A viable architecture needs compiled ROM/SRAM macros or chiplets and many more
matrix engines; synthesizing every model bit into standard-cell constants is
not a practical full-model implementation. The current tile proves that real
weights can become fixed gates and participate in inference.
