`timescale 1ns/1ps
`default_nettype none

module tb_q8_0_block_mac;
    logic clk = 0;
    logic rst = 1;
    logic input_valid = 0;
    logic input_ready;
    logic signed [255:0] weight_qs;
    logic signed [255:0] activation_qs;
    logic [15:0] weight_scale_fp16 = 16'h3800;     // 0.5
    logic [15:0] activation_scale_fp16 = 16'h3400; // 0.25
    logic output_valid;
    logic signed [31:0] integer_dot;
    logic [15:0] weight_scale_out;
    logic [15:0] activation_scale_out;
    integer i;

    always #5 clk = ~clk;

    q8_0_block_mac dut (.*);

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
        // 2 * sum(-16..15) = -32
        if (integer_dot !== -32) $fatal(1, "dot mismatch: %0d", integer_dot);
        if (weight_scale_out !== 16'h3800) $fatal(1, "weight scale mismatch");
        if (activation_scale_out !== 16'h3400) $fatal(1, "activation scale mismatch");
        $display("Q8_0 BLOCK MAC PASSED: integer dot=%0d", integer_dot);
        $finish;
    end
endmodule

`default_nettype wire
