# Tiny transformer ASIC macro

This directory turns the synthesizable `tiny_transformer` demonstration into
a pad-agnostic ASIC macro. It is a physical-design starting point, not a
fabrication order and not the exact GPT-2 golden model.

## Frozen proof-chip configuration

- vocabulary: 4
- sequence length: 2
- model width: 2
- feed-forward width: 4
- parameters: 70 FP32 words
- nominal clock: 10 MHz
- parameter interface: one 32-bit word per write
- inference latency in the zero-model regression: 63 cycles

The configuration bus replaces the original wide weight ports. Addresses 0
through 69 hold parameters in the order declared in
`tiny_transformer_asic_top.sv`; address 512 holds the two 8-bit token IDs.
Configuration writes are accepted only while `cfg_ready` is high.

## Verify RTL

```bash
cd asic
make sim
```

## Generate a generic synthesized netlist

```bash
cd asic
make synth
```

FP32 expansion is intentionally expensive and synthesis can take several
minutes. SAT resource sharing is disabled because its proof cost dominates
this datapath.

## Generate SKY130 layout

Install LibreLane, start Docker Desktop, then run:

```bash
python3 -m pip install librelane
cd asic
make gds
```

The LibreLane configuration is `sky130/config.json`. Successful completion
produces GDSII, DEF, LEF, timing, DRC and LVS results below `sky130/runs/`.
This FP32 design needs substantial host memory; synthesis and detailed routing
both approached 8 GB in the measured run. If LibreLane's synthesis resource
sharing stalls, the equivalent explicit mapping path is:

```bash
cd asic
make map-sky130
```

That target disables SAT-based sharing, maps the generic netlist to
`sky130_fd_sc_hd`, and writes the OpenROAD-readable netlist at
`build/tiny_transformer_asic_top_sky130_clean.v`. `SKY130_LIB` can be set to
the absolute path of a different SKY130 HD liberty file.

## Measured open-source-flow result

The current mapped prototype contains about 1.287 mm² of standard-cell area
inside a 3 mm by 3 mm die. Placement, PDN, clock-tree synthesis and global
routing complete. At the requested 10 MHz constraint, post-CTS vectorless
power is estimated at 1.109 W and timing does not close: setup WNS is
-73.705 ns and hold WNS is -0.552 ns. Detailed routing exceeded the Docker
memory allocation on the first ten-thread attempt. A one-thread retry stayed
within 6.3 GiB, but stalled in a congested region at 40% of its first pass with
14,336 violations; its log is retained in the local run directory.
`DRT_THREADS` is therefore limited to one in the checked-in configuration.
These numbers are engineering diagnostics, not signoff results and not a
tapeout-ready GDS.

The next silicon revision needs registered/pipelined FP32 arithmetic (or,
preferably for inference cost, INT8/INT4 arithmetic), SRAM/ROM weight storage,
high-fanout configuration buffering, timing repair, antenna repair, DRC, LVS,
IR-drop analysis and foundry signoff.

## Important architectural boundary

The core implemented here uses causal hardmax attention, ReLU and no
LayerNorm. It is the existing compact transformer demonstration. The
`gpt2_block_exact_sim.sv` module cannot be synthesized because it uses
`shortreal`, `$sqrt`, `$exp` and `$tanh` as a numerical oracle. Exact GPT-2
silicon requires replacing those functions with pipelined arithmetic blocks
and adding a latency-tolerant controller before fabrication.

For a cost-efficient production ASIC, the next revision should replace FP32
parameters and arithmetic with INT4/INT8 or another trained quantization
format, and replace the flip-flop parameter bank with ROM/SRAM macros.

## W4A8 copy

The separate `tiny_transformer_int4` implementation keeps the same transformer
graph and controller while using signed INT4 weights/embeddings, signed INT8
activations and INT32 accumulation/bias. The FP32 implementation is unchanged.
The small ASIC wrapper packs its 52 INT4 values eight per configuration word;
including 18 INT32 biases, the model contains 784 meaningful parameter bits in
27 configuration words. Run it with:

```bash
cd asic
make sim-int4
make synth-int4
make gds-int4
```

The generic default-size INT4 model uses 52,992 parameter bits, versus 341,504
bits for the FP32 version (about 6.4 times smaller). Values are currently
requantized with signed saturation and an implicit unit scale. Deploying real
trained weights requires calibration or quantization-aware training to supply
per-layer scales and correctly quantized INT32 biases.
