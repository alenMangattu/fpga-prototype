`timescale 1ns/1ps
`include "tile_config.svh"
`include "test_config.svh"
module tb_llama_frozen_tile;
    localparam integer BLOCKS = `LLAMA_TILE_COLUMNS / 32;
    localparam integer ROWS = `LLAMA_TILE_ROWS;
    localparam integer CASES = `LLAMA_TEST_CASES;
    reg clk=0, reset_n=0, start=0, act_we=0;
    reg [31:0] act_address=0;
    reg [271:0] act_data=0;
    wire start_ready, busy, done, result_valid;
    wire [31:0] result_row, result_fp32, cycle_count;
    reg [271:0] vectors [0:CASES*BLOCKS-1];
    reg [31:0] expected [0:CASES*ROWS-1];
    integer t, b, seen, elapsed, passes=0;
    llama_frozen_tile_top dut(.*);
    always #5 clk=~clk;

    task run_check(input integer vector_id);
        begin
            @(negedge clk); start=1;
            @(negedge clk); start=0;
            seen=0; elapsed=0;
            while (!done && elapsed < ROWS*BLOCKS+32) begin
                // Writes and extra starts while busy must be ignored.
                act_we=(elapsed == 3); act_address=0; act_data=0;
                start=(elapsed == 4);
                @(negedge clk);
                elapsed=elapsed+1;
                if (result_valid) begin
                    if (result_row !== seen || seen >= ROWS)
                        $fatal(1,"row order/count mismatch");
                    if (result_fp32 !== expected[vector_id*ROWS+seen])
                        $fatal(1,"vector=%0d row=%0d got=%h expected=%h", vector_id, seen,
                               result_fp32, expected[vector_id*ROWS+seen]);
                    seen=seen+1;
                end
            end
            act_we=0; start=0;
            if (!done || busy || seen != ROWS) $fatal(1,"timeout or incomplete output");
            $display("PASS vector=%0d rows=%0d compute_cycles=%0d",vector_id,seen,cycle_count);
            passes=passes+seen;
            @(negedge clk);
            if (done || result_valid) $fatal(1,"output must be a one-cycle pulse");
        end
    endtask

    initial begin
        $readmemh("activations.hex", vectors);
        $readmemh("expected.hex", expected);
        repeat (2) @(negedge clk);
        reset_n=1;
        // Incomplete activation storage must not launch computation.
        start=1;
        repeat (3) @(negedge clk);
        if (busy || start_ready) $fatal(1,"started with uninitialized activations");
        start=0;
        for (t=0; t<CASES; t=t+1) begin
            for (b=0; b<BLOCKS; b=b+1) begin
                act_we=1; act_address=b; act_data=vectors[t*BLOCKS+b];
                @(negedge clk);
            end
            act_we=0;
            #1;
            if (!start_ready) $fatal(1,"not ready after loading all blocks");
            run_check(t);
        end
        // Reuse the last vector without reloading activation memory.
        run_check(CASES-1);
        // Abort in flight and check pipeline valid bits are flushed by reset.
        start=1; @(negedge clk); start=0;
        repeat (5) @(negedge clk);
        reset_n=0; @(negedge clk); reset_n=1;
        repeat (12) begin
            @(negedge clk);
            if (busy || done || result_valid || start_ready) $fatal(1,"reset failed to flush state");
        end
        $display("FROZEN LLAMA TILE PASSED: %0d bit-exact FP32 row checks, replay, busy protection and reset.",passes);
        $finish;
    end
    initial begin
        #100000000;
        $fatal(1,"global timeout");
    end
endmodule
