# Run from asic/ (included by Makefile). Dependencies stay in the project venv.
LLAMA_PYTHON ?= ../.venv/bin/python
LLAMA_SOURCE ?= ../models/llama3.2-1b-int8
LLAMA_ROWS ?= 8
LLAMA_FIRST_ROW ?= 0
LLAMA_TENSOR ?= blk.0.attn_q.weight
LLAMA_BUILD = build/llama_frozen
LLAMA_GDS_BUILD = build/llama_gds_frozen
LLAMA_GDS_ROWS ?= 1
LLAMA_RTL = ../rtl/q8_0_dot_pipeline.sv ../rtl/int32_to_fp32.sv \
	../rtl/fp16_to_fp32.sv ../rtl/fp32_mul.sv ../rtl/fp32_add.sv \
	llama_frozen_tile_top.sv $(LLAMA_BUILD)/llama_tile_rom.sv

.PHONY: llama-freeze llama-vectors llama-sim llama-synth llama-netlist-sim llama-run llama-cosim llama-layout-preview llama-render-latest llama-gds llama-core-sim llama-core-synth

# Complete transformer-graph RTL regression.  This uses a deliberately small
# parameter set so every operation can be checked quickly; the same core is
# dimension-parameterized for the 16-layer Llama configuration.
llama-core-sim:
	mkdir -p build
	iverilog -g2012 -Wall -s tb_llama_decode_core \
		-o build/llama_decode_core_test ../rtl/llama_decode_core.sv \
		../tb/tb_llama_decode_core.sv
	vvp build/llama_decode_core_test

llama-core-synth:
	mkdir -p build
	yosys -Q -q -l build/llama_decode_core_synth.log -p \
		'read_verilog -sv -defer ../rtl/llama_decode_core.sv; chparam -set D_MODEL 4 -set D_FF 4 -set HEADS 2 -set KV_HEADS 1 -set LAYERS 2 -set VOCAB_SIZE 4 -set MAX_CONTEXT 2 llama_decode_core; hierarchy -top llama_decode_core; proc; opt; check -assert; memory -nomap; opt; check -assert; stat; write_verilog -noattr build/llama_decode_core_generic.v'

llama-freeze:
	$(LLAMA_PYTHON) ../tools/freeze_llama.py --source $(LLAMA_SOURCE) \
		--output $(LLAMA_BUILD) --rows $(LLAMA_ROWS) --first-row $(LLAMA_FIRST_ROW) --tile-tensor $(LLAMA_TENSOR)

llama-vectors: llama-freeze
	$(LLAMA_PYTHON) ../tools/test_llama_frozen.py --model $(LLAMA_BUILD)

llama-sim: llama-vectors
	iverilog -g2012 -s tb_q8_0_dot_pipeline -o $(LLAMA_BUILD)/dot_test \
		../rtl/q8_0_dot_pipeline.sv ../tb/tb_q8_0_dot_pipeline.sv
	vvp $(LLAMA_BUILD)/dot_test
	iverilog -g2012 -s tb_llama_frozen_tile -I $(LLAMA_BUILD) \
		-o $(LLAMA_BUILD)/rtl_test $(LLAMA_RTL) ../tb/tb_llama_frozen_tile.sv
	cd $(LLAMA_BUILD) && vvp rtl_test

llama-synth: llama-freeze
	yosys -Q -q -l $(LLAMA_BUILD)/synthesis.log -p \
		'read_verilog -sv -I$(LLAMA_BUILD) $(LLAMA_RTL); synth -top llama_frozen_tile_top -flatten -noshare; check -assert; stat; write_verilog -noattr $(LLAMA_BUILD)/tile_netlist.v; write_json $(LLAMA_BUILD)/tile_netlist.json'

llama-netlist-sim: llama-synth llama-vectors
	iverilog -g2012 -s tb_llama_frozen_tile -I $(LLAMA_BUILD) \
		-o $(LLAMA_BUILD)/netlist_test $(LLAMA_BUILD)/tile_netlist.v ../tb/tb_llama_frozen_tile.sv
	cd $(LLAMA_BUILD) && vvp netlist_test

llama-run: llama-freeze
	$(LLAMA_PYTHON) ../tools/llama_frozen_reference.py --model $(LLAMA_BUILD) \
		--report $(LLAMA_BUILD)/reference_run.json

llama-cosim: llama-freeze
	iverilog -g2012 -s tb_llama_frozen_infer -I $(LLAMA_BUILD) \
		-o $(LLAMA_BUILD)/infer_tile $(LLAMA_RTL) ../tb/tb_llama_frozen_infer.sv
	$(LLAMA_PYTHON) ../tools/llama_frozen_reference.py --model $(LLAMA_BUILD) \
		--rtl-tile $(LLAMA_BUILD)/infer_tile --report $(LLAMA_BUILD)/cosim_run.json

# Physical-design proof for the selected fixed-weight tile. This does not lay
# out the complete Llama decoder or its 1.31 GB model image.
llama-gds:
	$(LLAMA_PYTHON) ../tools/freeze_llama.py --source $(LLAMA_SOURCE) \
		--output $(LLAMA_GDS_BUILD) --rows $(LLAMA_GDS_ROWS) \
		--first-row $(LLAMA_FIRST_ROW) --tile-tensor $(LLAMA_TENSOR)
	$(LLAMA_PYTHON) -m librelane --docker-no-tty --dockerized sky130/config_llama_tile.json

# A bounded physical prototype that stops after clock-tree synthesis. This is
# useful when the standard-cell memory implementation cannot finish routing.
llama-layout-preview:
	$(LLAMA_PYTHON) ../tools/freeze_llama.py --source $(LLAMA_SOURCE) \
		--output $(LLAMA_GDS_BUILD) --rows $(LLAMA_GDS_ROWS) \
		--first-row $(LLAMA_FIRST_ROW) --tile-tensor $(LLAMA_TENSOR)
	$(LLAMA_PYTHON) -m librelane --docker-no-tty --dockerized \
		--to OpenROAD.CTS sky130/config_llama_tile.json
	$(MAKE) llama-render-latest

llama-render-latest:
	@run_dir=$$(ls -td sky130/runs/RUN_* | head -1); \
	def_file=$$run_dir/35-openroad-cts/llama_frozen_tile_top.def; \
	test -f $$def_file || { echo "CTS DEF not found in $$run_dir"; exit 1; }; \
	python3 scripts/render_def_svg.py $$def_file --out-dir $$run_dir/render; \
	echo "Rendered $$run_dir/render/llama_frozen_tile_top.overview.svg"
