RTL = ../rtl/fp32_mul.sv \
	../rtl/fp32_add.sv \
	../rtl/fp32_relu.sv \
	../rtl/mac.sv \
	../rtl/fp32_linear.sv \
	../rtl/tiny_transformer.sv \
	tiny_transformer_asic_top.sv

INT4_RTL = ../rtl/int4_linear.sv \
	../rtl/int8_dot_bank.sv \
	../rtl/tiny_transformer_int4.sv \
	tiny_transformer_int4_asic_top.sv

SKY130_LIB ?= $(firstword $(wildcard $(HOME)/.ciel/ciel/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib))

.PHONY: sim synth map-sky130 gds sim-int4 synth-int4 gds-int4 render-int4 clean

sim:
	mkdir -p build
	iverilog -g2012 -Wall -s tb_tiny_transformer_asic_top \
		-o build/asic_rtl_test $(RTL) ../tb/tb_tiny_transformer_asic_top.sv
	vvp build/asic_rtl_test

synth:
	mkdir -p build
	yosys -q -l build/yosys_generic.log -p \
		'read_verilog -sv $(RTL); synth -top tiny_transformer_asic_top -flatten -noshare -noabc; check; stat; write_verilog -noattr build/tiny_transformer_asic_top_generic.v'

map-sky130: synth
	@test -n "$(SKY130_LIB)" || (echo "SKY130 liberty file not found; set SKY130_LIB=/absolute/path/to/sky130_fd_sc_hd__tt_025C_1v80.lib"; exit 1)
	yosys -q -l build/yosys_sky130_map.log -p \
		'read_verilog build/tiny_transformer_asic_top_generic.v; proc; opt; memory_map; techmap; opt; dfflibmap -liberty $(SKY130_LIB); abc -liberty $(SKY130_LIB); clean; stat -liberty $(SKY130_LIB); write_verilog -noattr build/tiny_transformer_asic_top_sky130.v'
	yosys -q -l build/yosys_sky130_sanitize.log -p \
		'read_liberty -lib $(SKY130_LIB); read_verilog build/tiny_transformer_asic_top_sky130.v; hierarchy -top tiny_transformer_asic_top; rename -unescape; write_verilog -noattr -noexpr build/tiny_transformer_asic_top_sky130_clean.v; check'
	perl -pi -e 's/wire signed /wire /g' build/tiny_transformer_asic_top_sky130_clean.v
	yosys -q -l build/yosys_sky130_check.log -p \
		'read_liberty -lib $(SKY130_LIB); read_verilog build/tiny_transformer_asic_top_sky130_clean.v; hierarchy -top tiny_transformer_asic_top; check; stat -liberty $(SKY130_LIB)'

gds:
	python3 -m librelane --dockerized sky130/config.json

sim-int4:
	mkdir -p build
	iverilog -g2012 -Wall -s tb_tiny_transformer_int4 \
		-o build/int4_transformer_test $(INT4_RTL) \
		../tb/tb_tiny_transformer_int4.sv
	vvp build/int4_transformer_test
	iverilog -g2012 -Wall -s tb_tiny_transformer_int4_asic_top \
		-o build/int4_asic_rtl_test $(INT4_RTL) \
		../tb/tb_tiny_transformer_int4_asic_top.sv
	vvp build/int4_asic_rtl_test

synth-int4:
	mkdir -p build
	yosys -q -l build/yosys_int4.log -p \
		'read_verilog -sv $(INT4_RTL); synth -top tiny_transformer_int4_asic_top -flatten -noshare; check; stat; write_verilog -noattr build/tiny_transformer_int4_asic_top_generic.v'

gds-int4:
	python3 -m librelane --dockerized sky130/config_int4.json

render-int4:
	python3 scripts/render_def_svg.py \
		sky130/runs/int4_baseline/final/def/tiny_transformer_int4_asic_top.def \
		--out-dir sky130/runs/int4_baseline/final/render
	mkdir -p sky130/runs/int4_baseline/final/render/quicklook
	qlmanage -t -s 1800 -o sky130/runs/int4_baseline/final/render/quicklook \
		sky130/runs/int4_baseline/final/render/tiny_transformer_int4_asic_top.overview.svg \
		sky130/runs/int4_baseline/final/render/tiny_transformer_int4_asic_top.core.svg \
		sky130/runs/int4_baseline/final/render/tiny_transformer_int4_asic_top.routes_zoom.svg

clean:
	rm -rf build sky130/runs

include llama.mk
