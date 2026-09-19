`timescale 1ns/1ps
`default_nettype none

module tb_q8_0_block_fp32;
    logic clk = 0, rst = 1, input_valid = 0, input_ready, output_valid;
    logic signed [255:0] weight_qs, activation_qs;
    logic [15:0] weight_scale_fp16 = 16'h3800;     // 0.5
    logic [15:0] activation_scale_fp16 = 16'h3400; // 0.25
    logic [31:0] block_value;
    integer i;

    always #5 clk = ~clk;
    q8_0_block_fp32 dut (.*);

    initial begin
        weight_qs = '0;
        activation_qs = '0;
        for (i = 0; i < 32; i = i + 1) begin
            weight_qs[i*8 +: 8] = $signed(i - 16);
            activation_qs[i*8 +: 8] = $signed(2);
        end
        repeat (2) @(posedge clk);
        rst <= 0;
        input_valid <= 1;
        @(posedge clk);
        input_valid <= 0;
        @(posedge clk);
        if (!output_valid) $fatal(1, "missing output_valid");
        // -32 * 0.5 * 0.25 = -4.0
        if (block_value !== 32'hc0800000)
            $fatal(1, "scaled block mismatch: %h", block_value);
        $display("Q8_0 FP32 BLOCK PASSED: result=%h (-4.0)", block_value);
        $finish;
    end
endmodule

`default_nettype wire
