`timescale 1ns/1ps
`default_nettype none

// One-cycle registered Q8_0 block dot followed by combinational FP32 scaling.
// block_value = integer_dot * fp16(weight_scale) * fp16(activation_scale)
module q8_0_block_fp32 (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   input_valid,
    output logic                   input_ready,
    input  logic signed [255:0]    weight_qs,
    input  logic signed [255:0]    activation_qs,
    input  logic [15:0]            weight_scale_fp16,
    input  logic [15:0]            activation_scale_fp16,
    output logic                   output_valid,
    output logic [31:0]            block_value
);
    logic signed [31:0] integer_dot;
    logic [15:0] weight_scale_registered, activation_scale_registered;
    logic [31:0] integer_fp32, weight_scale_fp32, activation_scale_fp32;
    logic [31:0] scaled_once;

    q8_0_block_mac mac (
        .clk(clk), .rst(rst), .input_valid(input_valid),
        .input_ready(input_ready), .weight_qs(weight_qs),
        .activation_qs(activation_qs),
        .weight_scale_fp16(weight_scale_fp16),
        .activation_scale_fp16(activation_scale_fp16),
        .output_valid(output_valid), .integer_dot(integer_dot),
        .weight_scale_out(weight_scale_registered),
        .activation_scale_out(activation_scale_registered)
    );

    int32_to_fp32 integer_converter(.value(integer_dot), .result(integer_fp32));
    fp16_to_fp32 weight_converter(.value(weight_scale_registered), .result(weight_scale_fp32));
    fp16_to_fp32 activation_converter(.value(activation_scale_registered), .result(activation_scale_fp32));
    fp32_mul multiply_weight(.a(integer_fp32), .b(weight_scale_fp32), .result(scaled_once));
    fp32_mul multiply_activation(.a(scaled_once), .b(activation_scale_fp32), .result(block_value));
endmodule

`default_nettype wire
