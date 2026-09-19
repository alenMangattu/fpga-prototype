`timescale 1ns/1ps
`default_nettype none

module tb_fp32_matrix_mul;
    localparam integer M = 2;
    localparam integer K = 3;
    localparam integer N = 2;

    logic clk;
    logic rst;
    logic start;
    logic [M*K*32-1:0] matrix_a;
    logic [K*N*32-1:0] matrix_b;
    logic [M*N*32-1:0] matrix_c;
    logic valid;
    integer cycle;

    fp32_matrix_mul #(.M(M), .K(K), .N(N)) dut (
        .clk(clk), .rst(rst), .start(start),
        .matrix_a(matrix_a), .matrix_b(matrix_b),
        .matrix_c(matrix_c), .valid(valid)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (rst)
            cycle <= 0;
        else
            cycle <= cycle + 1;
    end

    task automatic set_a (
        input integer row,
        input integer column,
        input logic [31:0] value
    );
        matrix_a[(row*K + column)*32 +: 32] = value;
    endtask

    task automatic set_b (
        input integer row,
        input integer column,
        input logic [31:0] value
    );
        matrix_b[(row*N + column)*32 +: 32] = value;
    endtask

    task automatic check_c (
        input integer row,
        input integer column,
        input logic [31:0] expected,
        input string decimal_value
    );
        logic [31:0] actual;
        begin
            actual = matrix_c[(row*N + column)*32 +: 32];
            if (actual !== expected)
                $fatal(1, "C[%0d,%0d]: expected %08h, got %08h",
                       row, column, expected, actual);
            $display("  C[%0d,%0d] = %s [FP32 bits: %08h]",
                     row, column, decimal_value, actual);
        end
    endtask

    initial begin
        rst = 1;
        start = 0;
        matrix_a = 0;
        matrix_b = 0;
        cycle = 0;

        repeat (2) @(negedge clk);
        rst = 0;

        // A = [[1, 2, 3], [4, 5, 6]]
        set_a(0, 0, 32'h3f80_0000);
        set_a(0, 1, 32'h4000_0000);
        set_a(0, 2, 32'h4040_0000);
        set_a(1, 0, 32'h4080_0000);
        set_a(1, 1, 32'h40a0_0000);
        set_a(1, 2, 32'h40c0_0000);

        // B = [[7, 8], [9, 10], [11, 12]]
        set_b(0, 0, 32'h40e0_0000);
        set_b(0, 1, 32'h4100_0000);
        set_b(1, 0, 32'h4110_0000);
        set_b(1, 1, 32'h4120_0000);
        set_b(2, 0, 32'h4130_0000);
        set_b(2, 1, 32'h4140_0000);

        $display("A (2x3) * B (3x2):");
        $display("  C[0,0] = 1*7 + 2*9 + 3*11");
        $display("  C[0,1] = 1*8 + 2*10 + 3*12");
        $display("  C[1,0] = 4*7 + 5*9 + 6*11");
        $display("  C[1,1] = 4*8 + 5*10 + 6*12");

        @(negedge clk);
        start = 1;
        $display("START cycle %0d: all four dot products run in parallel", cycle);
        @(posedge clk);
        #1;
        start = 0;

        if (!valid)
            $fatal(1, "Matrix result was not valid after one clock");

        $display("DONE  cycle %0d: complete result matrix", cycle);
        check_c(0, 0, 32'h4268_0000, "58.0");
        check_c(0, 1, 32'h4280_0000, "64.0");
        check_c(1, 0, 32'h430b_0000, "139.0");
        check_c(1, 1, 32'h431a_0000, "154.0");

        @(posedge clk);
        #1;
        if (valid)
            $fatal(1, "valid did not return low");

        $display("ALL FP32 MATRIX-MULTIPLICATION TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
