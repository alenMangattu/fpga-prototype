`timescale 1ns/1ps
`default_nettype none

// MNIST-sized FP32 neural-network accelerator:
//
//   784 pixels -> 100 hidden neurons + ReLU -> 10 output scores
//
// One MAC is instantiated per neuron. All 100 hidden neurons process the same
// pixel concurrently for 784 clocks, then all 10 output neurons process the
// same hidden activation concurrently for 100 clocks.
module fp32_mnist #(
    parameter integer INPUT_SIZE  = 784,
    parameter integer HIDDEN_SIZE = 100,
    parameter integer OUTPUT_SIZE = 10
) (
    input  logic                                  clk,
    input  logic                                  rst,
    input  logic                                  start,
    input  logic [INPUT_SIZE*32-1:0]              pixels,
    input  logic [HIDDEN_SIZE*INPUT_SIZE*32-1:0]  weights_1,
    input  logic [HIDDEN_SIZE*32-1:0]             bias_1,
    input  logic [OUTPUT_SIZE*HIDDEN_SIZE*32-1:0] weights_2,
    input  logic [OUTPUT_SIZE*32-1:0]             bias_2,
    output logic [HIDDEN_SIZE*32-1:0]             hidden_activations,
    output logic [OUTPUT_SIZE*32-1:0]             scores,
    output logic [$clog2(OUTPUT_SIZE)-1:0]        predicted_class,
    output logic                                  busy,
    output logic                                  done,
    output logic [31:0]                           inference_cycles
);
    localparam logic [2:0] IDLE         = 3'd0;
    localparam logic [2:0] CLEAR_HIDDEN = 3'd1;
    localparam logic [2:0] HIDDEN       = 3'd2;
    localparam logic [2:0] CLEAR_OUTPUT = 3'd3;
    localparam logic [2:0] OUTPUT_LAYER = 3'd4;
    localparam logic [2:0] DONE_STATE   = 3'd5;

    localparam integer INPUT_INDEX_WIDTH = $clog2(INPUT_SIZE);
    localparam integer HIDDEN_INDEX_WIDTH = $clog2(HIDDEN_SIZE);
    localparam logic [INPUT_INDEX_WIDTH-1:0] LAST_INPUT =
        INPUT_INDEX_WIDTH'(INPUT_SIZE - 1);
    localparam logic [HIDDEN_INDEX_WIDTH-1:0] LAST_HIDDEN =
        HIDDEN_INDEX_WIDTH'(HIDDEN_SIZE - 1);

    logic [2:0] state;
    logic [INPUT_INDEX_WIDTH-1:0] input_index;
    logic [HIDDEN_INDEX_WIDTH-1:0] hidden_index;

    logic [31:0] current_pixel;
    logic [31:0] current_hidden;
    logic [31:0] hidden_weight [0:HIDDEN_SIZE-1];
    logic [31:0] hidden_acc    [0:HIDDEN_SIZE-1];
    logic [31:0] hidden_biased [0:HIDDEN_SIZE-1];
    logic [31:0] hidden_relu   [0:HIDDEN_SIZE-1];
    logic [31:0] output_weight [0:OUTPUT_SIZE-1];
    logic [31:0] output_acc    [0:OUTPUT_SIZE-1];
    logic [31:0] output_biased [0:OUTPUT_SIZE-1];
    logic [31:0] largest_score;
    integer class_index;

    function automatic fp32_greater_than;
        input logic [31:0] a;
        input logic [31:0] b;
        logic a_nan;
        logic b_nan;
        begin
            a_nan = (a[30:23] == 8'hff) && (a[22:0] != 0);
            b_nan = (b[30:23] == 8'hff) && (b[22:0] != 0);
            if (a_nan)
                fp32_greater_than = 1'b0;
            else if (b_nan)
                fp32_greater_than = 1'b1;
            else if ((a[30:0] == 0) && (b[30:0] == 0))
                fp32_greater_than = 1'b0;
            else if (a[31] != b[31])
                fp32_greater_than = !a[31];
            else if (!a[31])
                fp32_greater_than = a[30:0] > b[30:0];
            else
                fp32_greater_than = a[30:0] < b[30:0];
        end
    endfunction

    assign current_pixel = pixels[input_index*32 +: 32];
    assign current_hidden = hidden_activations[hidden_index*32 +: 32];

    genvar h;
    generate
        for (h = 0; h < HIDDEN_SIZE; h = h + 1) begin : hidden_neuron
            assign hidden_weight[h] =
                weights_1[(h*INPUT_SIZE + 32'(input_index))*32 +: 32];

            mac neuron_mac (
                .clk    (clk),
                .rst    (rst),
                .clear  (state == CLEAR_HIDDEN),
                .enable (state == HIDDEN),
                .a      (current_pixel),
                .b      (hidden_weight[h]),
                .acc    (hidden_acc[h])
            );

            fp32_add add_bias (
                .a      (hidden_acc[h]),
                .b      (bias_1[h*32 +: 32]),
                .result (hidden_biased[h])
            );

            fp32_relu activation (
                .value  (hidden_biased[h]),
                .result (hidden_relu[h])
            );

            assign hidden_activations[h*32 +: 32] = hidden_relu[h];
        end
    endgenerate

    genvar o;
    generate
        for (o = 0; o < OUTPUT_SIZE; o = o + 1) begin : output_neuron
            assign output_weight[o] =
                weights_2[(o*HIDDEN_SIZE + 32'(hidden_index))*32 +: 32];

            mac neuron_mac (
                .clk    (clk),
                .rst    (rst),
                .clear  (state == CLEAR_OUTPUT),
                .enable (state == OUTPUT_LAYER),
                .a      (current_hidden),
                .b      (output_weight[o]),
                .acc    (output_acc[o])
            );

            fp32_add add_bias (
                .a      (output_acc[o]),
                .b      (bias_2[o*32 +: 32]),
                .result (output_biased[o])
            );

            assign scores[o*32 +: 32] = output_biased[o];
        end
    endgenerate

    // Select the class with the largest FP32 output score.
    always @* begin
        largest_score = scores[0 +: 32];
        predicted_class = 0;
        for (class_index = 1; class_index < OUTPUT_SIZE;
             class_index = class_index + 1) begin
            if (fp32_greater_than(scores[class_index*32 +: 32], largest_score)) begin
                largest_score = scores[class_index*32 +: 32];
                predicted_class = $clog2(OUTPUT_SIZE)'(class_index);
            end
        end
    end

    always @* begin
        busy = (state != IDLE) && (state != DONE_STATE);
        done = (state == DONE_STATE);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            input_index <= 0;
            hidden_index <= 0;
            inference_cycles <= 0;
        end else begin
            case (state)
                IDLE: begin
                    input_index <= 0;
                    hidden_index <= 0;
                    if (start) begin
                        inference_cycles <= 0;
                        state <= CLEAR_HIDDEN;
                    end
                end

                CLEAR_HIDDEN: begin
                    inference_cycles <= inference_cycles + 1;
                    input_index <= 0;
                    state <= HIDDEN;
                end

                HIDDEN: begin
                    inference_cycles <= inference_cycles + 1;
                    if (input_index == LAST_INPUT) begin
                        input_index <= 0;
                        hidden_index <= 0;
                        state <= CLEAR_OUTPUT;
                    end else begin
                        input_index <= input_index + 1'b1;
                    end
                end

                CLEAR_OUTPUT: begin
                    inference_cycles <= inference_cycles + 1;
                    hidden_index <= 0;
                    state <= OUTPUT_LAYER;
                end

                OUTPUT_LAYER: begin
                    inference_cycles <= inference_cycles + 1;
                    if (hidden_index == LAST_HIDDEN) begin
                        hidden_index <= 0;
                        state <= DONE_STATE;
                    end else begin
                        hidden_index <= hidden_index + 1'b1;
                    end
                end

                DONE_STATE: begin
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
