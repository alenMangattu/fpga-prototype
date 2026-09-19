`timescale 1ns/1ps
`default_nettype none

module tb_mac;
    logic clk;
    logic rst;
    logic clear;
    logic enable;
    logic [31:0] a;
    logic [31:0] b;
    logic [31:0] acc;

    mac dut (
        .clk(clk), .rst(rst), .clear(clear), .enable(enable),
        .a(a), .b(b), .acc(acc)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    task automatic apply_and_check (
        input logic [31:0] operand_a,
        input logic [31:0] operand_b,
        input logic [31:0] expected,
        input string description
    );
        begin
            @(negedge clk);
            a = operand_a;
            b = operand_b;
            enable = 1;
            clear = 0;
            @(posedge clk);
            #1;
            if (acc !== expected)
                $fatal(1, "%s: expected %08h, got %08h",
                       description, expected, acc);
            $display("PASS: %s  [FP32 bits: %08h]", description, acc);
        end
    endtask

    task automatic clear_accumulator;
        begin
            @(negedge clk);
            enable = 0;
            clear = 1;
            @(posedge clk);
            #1;
            if (acc !== 32'h0000_0000)
                $fatal(1, "Clear failed: got %08h", acc);
            $display("CTRL: clear accumulator -> 0.0  [FP32 bits: %08h]", acc);
            clear = 0;
        end
    endtask

    initial begin
        rst = 1;
        clear = 0;
        enable = 0;
        a = 0;
        b = 0;

        repeat (2) @(negedge clk);
        rst = 0;

        // 0 + (3.0 * -4.0) = -12.0
        apply_and_check(32'h4040_0000, 32'hc080_0000,
                        32'hc140_0000, "0.0 + (3.0 * -4.0) = -12.0");

        // -12.0 + (-2.0 * 5.0) = -22.0
        apply_and_check(32'hc000_0000, 32'h40a0_0000,
                        32'hc1b0_0000, "-12.0 + (-2.0 * 5.0) = -22.0");

        clear_accumulator();

        $display("\n--- Dot product: [1.5, -2.25, 0.5] dot [4.0, 3.0, -8.0] ---");
        apply_and_check(32'h3fc0_0000, 32'h4080_0000,
                        32'h40c0_0000, "0.0 + (1.5 * 4.0) = 6.0");
        apply_and_check(32'hc010_0000, 32'h4040_0000,
                        32'hbf40_0000, "6.0 + (-2.25 * 3.0) = -0.75");
        apply_and_check(32'h3f00_0000, 32'hc100_0000,
                        32'hc098_0000, "-0.75 + (0.5 * -8.0) = -4.75");

        clear_accumulator();

        $display("\n--- Sum of squares: 1.25^2 + (-2.5)^2 + 0.75^2 ---");
        apply_and_check(32'h3fa0_0000, 32'h3fa0_0000,
                        32'h3fc8_0000, "0.0 + (1.25 * 1.25) = 1.5625");
        apply_and_check(32'hc020_0000, 32'hc020_0000,
                        32'h40fa_0000, "1.5625 + (-2.5 * -2.5) = 7.8125");
        apply_and_check(32'h3f40_0000, 32'h3f40_0000,
                        32'h4106_0000, "7.8125 + (0.75 * 0.75) = 8.375");

        clear_accumulator();

        $display("\n--- Polynomial at x=1.25: 2*x^2 - 3*x + 0.5 ---");
        apply_and_check(32'h4000_0000, 32'h3fc8_0000,
                        32'h4048_0000, "0.0 + (2.0 * 1.5625) = 3.125");
        apply_and_check(32'hc040_0000, 32'h3fa0_0000,
                        32'hbf20_0000, "3.125 + (-3.0 * 1.25) = -0.625");
        apply_and_check(32'h3f00_0000, 32'h3f80_0000,
                        32'hbe00_0000, "-0.625 + (0.5 * 1.0) = -0.125");

        clear_accumulator();

        $display("\n--- Floating-point order: large values cancel first ---");
        apply_and_check(32'h60ad_78ec, 32'h3f80_0000,
                        32'h60ad_78ec, "0.0 + (1e20 * 1.0) = 1e20");
        apply_and_check(32'he0ad_78ec, 32'h3f80_0000,
                        32'h0000_0000, "1e20 + (-1e20 * 1.0) = 0.0");
        apply_and_check(32'h4050_0000, 32'h3f80_0000,
                        32'h4050_0000, "0.0 + (3.25 * 1.0) = 3.25");

        clear_accumulator();

        $display("\n--- Same values, different order: the 3.25 is rounded away ---");
        apply_and_check(32'h60ad_78ec, 32'h3f80_0000,
                        32'h60ad_78ec, "0.0 + (1e20 * 1.0) = 1e20");
        apply_and_check(32'h4050_0000, 32'h3f80_0000,
                        32'h60ad_78ec, "1e20 + (3.25 * 1.0) rounds to 1e20");
        apply_and_check(32'he0ad_78ec, 32'h3f80_0000,
                        32'h0000_0000, "1e20 + (-1e20 * 1.0) = 0.0");

        clear_accumulator();

        // The multiplier must round 0.1 * 0.2 correctly to binary32.
        apply_and_check(32'h3dcc_cccd, 32'h3e4c_cccd,
                        32'h3ca3_d70b,
                        "0.0 + (0.1 * 0.2) = 0.020000001 (rounded)");

        clear_accumulator();

        // Smallest subnormal * 1.0 remains the smallest subnormal.
        apply_and_check(32'h0000_0001, 32'h3f80_0000,
                        32'h0000_0001,
                        "0.0 + (smallest subnormal * 1.0) = smallest subnormal");

        clear_accumulator();

        // Maximum finite number * 2.0 overflows to positive infinity.
        apply_and_check(32'h7f7f_ffff, 32'h4000_0000,
                        32'h7f80_0000,
                        "0.0 + (maximum finite * 2.0) = +infinity");

        clear_accumulator();

        // Infinity * zero is invalid and produces the canonical quiet NaN.
        apply_and_check(32'h7f80_0000, 32'h0000_0000,
                        32'h7fc0_0000,
                        "0.0 + (+infinity * 0.0) = NaN");

        clear_accumulator();

        // The accumulator must hold its exact bit pattern while disabled.
        @(negedge clk);
        a = 32'h42f6_0000; // 123.0
        b = 32'h42f6_0000;
        enable = 0;
        @(posedge clk);
        #1;
        if (acc !== 32'h0000_0000)
            $fatal(1, "Disabled MAC did not hold its value");
        $display("CTRL: enable=0, accumulator holds 0.0");

        // Reset has priority over clear and enable.
        @(negedge clk);
        rst = 1;
        clear = 1;
        enable = 1;
        @(posedge clk);
        #1;
        if (acc !== 32'h0000_0000)
            $fatal(1, "Reset priority failed");
        $display("CTRL: reset accumulator -> 0.0");

        $display("ALL FP32 TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
