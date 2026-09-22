`timescale 1ns/1ps
`include "tile_config.svh"
// One activation vector in, one fixed-weight tile result out for co-simulation.
module tb_llama_frozen_infer;
    localparam integer BLOCKS = `LLAMA_TILE_COLUMNS / 32;
    localparam integer ROWS = `LLAMA_TILE_ROWS;
    reg clk=0, reset_n=0, start=0, act_we=0;
    reg [31:0] act_address=0;
    reg [271:0] act_data=0;
    wire start_ready, busy, done, result_valid;
    wire [31:0] result_row, result_fp32, cycle_count;
    reg [271:0] activations [0:BLOCKS-1];
    integer b, seen=0, fd;
    llama_frozen_tile_top dut(.*);
    always #5 clk=~clk;
    initial begin
        $readmemh("activations.hex", activations);
        fd=$fopen("result.hex", "w");
        if (!fd) $fatal(1,"cannot open result.hex");
        repeat (2) @(negedge clk);
        reset_n=1;
        for (b=0; b<BLOCKS; b=b+1) begin
            act_we=1; act_address=b; act_data=activations[b];
            @(negedge clk);
        end
        act_we=0; #1;
        if (!start_ready) $fatal(1,"activation loading failed");
        start=1; @(negedge clk); start=0;
        while (!done) begin
            @(negedge clk);
            if (result_valid) begin
                if (result_row !== seen) $fatal(1,"row order mismatch");
                $fdisplay(fd,"%08h",result_fp32);
                seen=seen+1;
            end
        end
        if (seen != ROWS) $fatal(1,"incomplete output");
        $fclose(fd);
        $display("ROM_SHA256=%064h CYCLES=%0d", `LLAMA_TILE_SHA256, cycle_count);
        $finish;
    end
    initial begin
        #(10*(ROWS*BLOCKS+BLOCKS+100));
        $fatal(1,"tile inference timeout");
    end
endmodule
