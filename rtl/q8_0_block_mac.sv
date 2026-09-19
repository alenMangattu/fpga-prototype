`timescale 1ns/1ps
`default_nettype none

// Dot product of one GGML Q8_0 block. Q8_0 stores 32 signed weights and one
// FP16 scale. Activations are supplied as a separately quantized signed Q8
// block. The integer accumulator is exact; the two scales are applied by the
// surrounding floating-point pipeline.
module q8_0_block_mac (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    input_valid,
    output logic                    input_ready,
    input  logic signed [8*32-1:0]  weight_qs,
    input  logic signed [8*32-1:0]  activation_qs,
    input  logic [15:0]             weight_scale_fp16,
    input  logic [15:0]             activation_scale_fp16,
    output logic                    output_valid,
    output logic signed [31:0]      integer_dot,
    output logic [15:0]             weight_scale_out,
    output logic [15:0]             activation_scale_out
);
    integer lane;
    logic signed [31:0] sum;

    assign input_ready = 1'b1;

    always_comb begin
        sum = 32'sd0;
        for (lane = 0; lane < 32; lane = lane + 1)
            sum = sum
                + $signed(weight_qs[lane*8 +: 8])
                * $signed(activation_qs[lane*8 +: 8]);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            output_valid <= 1'b0;
            integer_dot <= 32'sd0;
            weight_scale_out <= 16'h0000;
            activation_scale_out <= 16'h0000;
        end else begin
            output_valid <= input_valid;
            if (input_valid) begin
                integer_dot <= sum;
                weight_scale_out <= weight_scale_fp16;
                activation_scale_out <= activation_scale_fp16;
            end
        end
    end
endmodule

`default_nettype wire
