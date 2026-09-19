`timescale 1ns/1ps
`default_nettype none

// Vendor-neutral ready/valid boundary for the nonlinear FP32 operations used
// by an exact GPT-2 block. OP: 0=divide, 1=sqrt, 2=exp, 3=tanh.
//
// Simulation uses IEEE shortreal system functions. For FPGA synthesis define
// SYNTHESIS and replace fp32_math_vendor with generated Efinix/AMD IP while
// preserving this handshake.
module fp32_math_ip #(
    parameter integer OP = 0
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        input_valid,
    output logic        input_ready,
    input  logic [31:0] operand_a,
    input  logic [31:0] operand_b,
    output logic        output_valid,
    output logic [31:0] result
);
`ifndef SYNTHESIS
    shortreal a_value, b_value, result_value;
    assign input_ready = 1'b1;

    always_ff @(posedge clk) begin
        if (rst) begin
            output_valid <= 1'b0;
            result <= 32'h0000_0000;
        end else begin
            output_valid <= input_valid;
            if (input_valid) begin
                a_value = $bitstoshortreal(operand_a);
                b_value = $bitstoshortreal(operand_b);
                case (OP)
                    0: result_value = a_value / b_value;
                    1: result_value = $sqrt(a_value);
                    2: result_value = $exp(a_value);
                    3: result_value = $tanh(a_value);
                    default: result_value = 0.0;
                endcase
                result <= $shortrealtobits(result_value);
            end
        end
    end
`else
    fp32_math_vendor #(.OP(OP)) vendor_core (
        .clk(clk), .rst(rst),
        .input_valid(input_valid), .input_ready(input_ready),
        .operand_a(operand_a), .operand_b(operand_b),
        .output_valid(output_valid), .result(result)
    );
`endif
endmodule

// Synthesis black box to be replaced by the selected FPGA's generated cores.
(* blackbox *)
module fp32_math_vendor #(
    parameter integer OP = 0
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        input_valid,
    output logic        input_ready,
    input  logic [31:0] operand_a,
    input  logic [31:0] operand_b,
    output logic        output_valid,
    output logic [31:0] result
);
endmodule

`default_nettype wire
