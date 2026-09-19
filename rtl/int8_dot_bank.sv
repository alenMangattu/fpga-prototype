`timescale 1ns/1ps
`default_nettype none

// Parallel bank of signed INT8 dot products used for attention Q*K scores.
module int8_dot_bank #(
    parameter integer IN_SIZE  = 4,
    parameter integer OUT_SIZE = 4
) (
    input  logic                            clk,
    input  logic                            rst,
    input  logic                            start,
    input  logic [IN_SIZE*8-1:0]            input_vector,
    input  logic [OUT_SIZE*IN_SIZE*8-1:0]   vectors,
    output logic [OUT_SIZE*32-1:0]          scores,
    output logic                            busy,
    output logic                            done
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
    logic signed [7:0] input_value;
    logic signed [31:0] accumulator [0:OUT_SIZE-1];

    assign input_value = $signed(input_vector[input_index*8 +: 8]);
    assign busy = (state != IDLE) && (state != FINISHED);
    assign done = (state == FINISHED);

    genvar output_index;
    generate
        for (output_index = 0; output_index < OUT_SIZE;
             output_index = output_index + 1) begin : output_lane
            logic signed [7:0] vector_value;
            logic signed [15:0] product;

            assign vector_value = $signed(vectors[
                (output_index*IN_SIZE + input_index)*8 +: 8]);
            assign product = input_value * vector_value;
            assign scores[output_index*32 +: 32] =
                accumulator[output_index];

            always_ff @(posedge clk) begin
                if (rst || state == CLEAR)
                    accumulator[output_index] <= 0;
                else if (state == COMPUTE)
                    accumulator[output_index] <=
                        accumulator[output_index] + product;
            end
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
                CLEAR: state <= COMPUTE;
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
