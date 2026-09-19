`timescale 1ns/1ps
`default_nettype none

module tb_fp32_math_ip;
    logic clk, rst, input_valid;
    logic [31:0] div_a, div_b, sqrt_a, exp_a, tanh_a;
    logic div_valid, sqrt_valid, exp_valid, tanh_valid;
    logic [31:0] div_result, sqrt_result, exp_result, tanh_result;

    fp32_math_ip #(.OP(0)) divide (
        .clk(clk),.rst(rst),.input_valid(input_valid),.input_ready(),
        .operand_a(div_a),.operand_b(div_b),
        .output_valid(div_valid),.result(div_result));
    fp32_math_ip #(.OP(1)) square_root (
        .clk(clk),.rst(rst),.input_valid(input_valid),.input_ready(),
        .operand_a(sqrt_a),.operand_b(0),
        .output_valid(sqrt_valid),.result(sqrt_result));
    fp32_math_ip #(.OP(2)) exponential (
        .clk(clk),.rst(rst),.input_valid(input_valid),.input_ready(),
        .operand_a(exp_a),.operand_b(0),
        .output_valid(exp_valid),.result(exp_result));
    fp32_math_ip #(.OP(3)) hyperbolic_tangent (
        .clk(clk),.rst(rst),.input_valid(input_valid),.input_ready(),
        .operand_a(tanh_a),.operand_b(0),
        .output_valid(tanh_valid),.result(tanh_result));

    initial begin clk=0; forever #5 clk=~clk; end
    initial begin
        rst=1; input_valid=0;
        div_a=32'h40c0_0000; div_b=32'h4000_0000; // 6/2
        sqrt_a=32'h4080_0000; // sqrt(4)
        exp_a=32'h0000_0000;  // exp(0)
        tanh_a=32'h0000_0000; // tanh(0)
        repeat(2) @(negedge clk); rst=0;
        @(negedge clk); input_valid=1;
        @(negedge clk); input_valid=0;
        wait(div_valid && sqrt_valid && exp_valid && tanh_valid); #1;
        if(div_result!==32'h4040_0000) $fatal(1,"divide failed");
        if(sqrt_result!==32'h4000_0000) $fatal(1,"sqrt failed");
        if(exp_result!==32'h3f80_0000) $fatal(1,"exp failed");
        if(tanh_result!==32'h0000_0000) $fatal(1,"tanh failed");
        $display("FP32 MATH IP INTERFACE TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
