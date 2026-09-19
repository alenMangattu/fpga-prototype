`timescale 1ns/1ps
`default_nettype none

// IEEE-754 binary32 multiply-accumulate unit. Inputs and output are raw FP32
// bit patterns. Multiplication and addition each use round-to-nearest-even.
module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic        clear,
    input  logic        enable,
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] acc
);
    logic [31:0] product;
    logic [31:0] next_acc;

    fp32_mul multiplier (.a(a), .b(b), .result(product));
    fp32_add accumulator_adder (.a(acc), .b(product), .result(next_acc));

    always_ff @(posedge clk) begin
        if (rst)
            acc <= 32'h0000_0000;
        else if (clear)
            acc <= 32'h0000_0000;
        else if (enable)
            acc <= next_acc;
    end
endmodule

`default_nettype wire
