`timescale 1ns/1ps
`default_nettype none

// Fully parallel two-layer FP32 neural network:
//
//   hidden = ReLU(weights_1 * input + bias_1)
//   output = weights_2 * hidden + bias_2
//
// The design has two registered matrix-vector stages. First-result latency is
// two clocks; because both stages operate concurrently, throughput is one
// inference per clock when start is asserted on consecutive cycles.
module fp32_dnn #(
    parameter integer INPUT_SIZE  = 4,
    parameter integer HIDDEN_SIZE = 3,
    parameter integer OUTPUT_SIZE = 2
) (
    input  logic                                  clk,
    input  logic                                  rst,
    input  logic                                  start,
    input  logic [INPUT_SIZE*32-1:0]              input_vector,
    input  logic [HIDDEN_SIZE*INPUT_SIZE*32-1:0]  weights_1,
    input  logic [HIDDEN_SIZE*32-1:0]             bias_1,
    input  logic [OUTPUT_SIZE*HIDDEN_SIZE*32-1:0] weights_2,
    input  logic [OUTPUT_SIZE*32-1:0]             bias_2,
    output logic [HIDDEN_SIZE*32-1:0]             hidden_vector,
    output logic [OUTPUT_SIZE*32-1:0]             output_vector,
    output logic                                  valid
);
    logic [31:0] hidden_dot    [0:HIDDEN_SIZE-1];
    logic [31:0] hidden_biased [0:HIDDEN_SIZE-1];
    logic [31:0] hidden_relu   [0:HIDDEN_SIZE-1];
    logic        hidden_valid  [0:HIDDEN_SIZE-1];

    logic [31:0] output_dot    [0:OUTPUT_SIZE-1];
    logic [31:0] output_biased [0:OUTPUT_SIZE-1];
    logic        output_valid  [0:OUTPUT_SIZE-1];

    genvar hidden;
    generate
        for (hidden = 0; hidden < HIDDEN_SIZE; hidden = hidden + 1) begin : layer_1
            fp32_dot_product #(.LENGTH(INPUT_SIZE)) neuron_dot (
                .clk      (clk),
                .rst      (rst),
                .start    (start),
                .vector_a (input_vector),
                .vector_b (weights_1[hidden*INPUT_SIZE*32 +: INPUT_SIZE*32]),
                .result   (hidden_dot[hidden]),
                .valid    (hidden_valid[hidden])
            );

            fp32_add add_bias (
                .a      (hidden_dot[hidden]),
                .b      (bias_1[hidden*32 +: 32]),
                .result (hidden_biased[hidden])
            );

            fp32_relu activation (
                .value  (hidden_biased[hidden]),
                .result (hidden_relu[hidden])
            );

            assign hidden_vector[hidden*32 +: 32] = hidden_relu[hidden];
        end
    endgenerate

    genvar output_index;
    generate
        for (output_index = 0; output_index < OUTPUT_SIZE;
             output_index = output_index + 1) begin : layer_2
            fp32_dot_product #(.LENGTH(HIDDEN_SIZE)) neuron_dot (
                .clk      (clk),
                .rst      (rst),
                .start    (hidden_valid[0]),
                .vector_a (hidden_vector),
                .vector_b (weights_2[output_index*HIDDEN_SIZE*32 +:
                                     HIDDEN_SIZE*32]),
                .result   (output_dot[output_index]),
                .valid    (output_valid[output_index])
            );

            fp32_add add_bias (
                .a      (output_dot[output_index]),
                .b      (bias_2[output_index*32 +: 32]),
                .result (output_biased[output_index])
            );

            assign output_vector[output_index*32 +: 32] =
                output_biased[output_index];
        end
    endgenerate

    assign valid = output_valid[0];
endmodule

`default_nettype wire
