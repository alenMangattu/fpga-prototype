`timescale 1ns/1ps
`default_nettype none

// FP32 linear layer: output = weights * input + bias.
// One MAC is used per output, so all outputs are computed in parallel while
// the input dimension is traversed over IN_SIZE clocks.
module fp32_linear #(
    parameter integer IN_SIZE  = 4,
    parameter integer OUT_SIZE = 4
) (
    input  logic                              clk,
    input  logic                              rst,
    input  logic                              start,
    input  logic [IN_SIZE*32-1:0]             input_vector,
    input  logic [OUT_SIZE*IN_SIZE*32-1:0]    weights,
    input  logic [OUT_SIZE*32-1:0]            bias,
    output logic [OUT_SIZE*32-1:0]            output_vector,
    output logic                              busy,
    output logic                              done
);
    localparam integer INDEX_WIDTH = (IN_SIZE <= 1) ? 1 : $clog2(IN_SIZE);
    localparam logic [1:0] IDLE = 2'd0;
    localparam logic [1:0] CLEAR = 2'd1;
    localparam logic [1:0] COMPUTE = 2'd2;
    localparam logic [1:0] FINISHED = 2'd3;
    localparam logic [INDEX_WIDTH-1:0] LAST_INDEX =
        INDEX_WIDTH'(IN_SIZE - 1);

    logic [1:0] state;
    logic [INDEX_WIDTH-1:0] input_index;
    logic [31:0] input_value;
    logic [31:0] accumulator [0:OUT_SIZE-1];
    logic [31:0] selected_weight [0:OUT_SIZE-1];

    assign input_value = input_vector[input_index*32 +: 32];
    assign busy = (state != IDLE) && (state != FINISHED);
    assign done = (state == FINISHED);

    genvar output_index;
    generate
        for (output_index = 0; output_index < OUT_SIZE;
             output_index = output_index + 1) begin : output_lane
            assign selected_weight[output_index] =
                weights[(output_index*IN_SIZE + 32'(input_index))*32 +: 32];

            mac dot_accumulator (
                .clk    (clk),
                .rst    (rst),
                .clear  (state == CLEAR),
                .enable (state == COMPUTE),
                .a      (input_value),
                .b      (selected_weight[output_index]),
                .acc    (accumulator[output_index])
            );

            fp32_add add_bias (
                .a      (accumulator[output_index]),
                .b      (bias[output_index*32 +: 32]),
                .result (output_vector[output_index*32 +: 32])
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            input_index <= 0;
        end else begin
            case (state)
                IDLE: begin
                    input_index <= 0;
                    if (start)
                        state <= CLEAR;
                end
                CLEAR: begin
                    input_index <= 0;
                    state <= COMPUTE;
                end
                COMPUTE: begin
                    if (input_index == LAST_INDEX) begin
                        input_index <= 0;
                        state <= FINISHED;
                    end else begin
                        input_index <= input_index + 1'b1;
                    end
                end
                FINISHED: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
