`timescale 1ns/1ps
`default_nettype none

module tb_fp32_dot16;
    logic clk;
    logic rst;
    logic start;
    logic [511:0] vector_a;
    logic [511:0] vector_b;
    logic [31:0] result;
    logic valid;
    integer cycle;

    fp32_dot16 dut (
        .clk(clk), .rst(rst), .start(start),
        .vector_a(vector_a), .vector_b(vector_b),
        .result(result), .valid(valid)
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

    task automatic set_element (
        input integer index,
        input logic [31:0] value_a,
        input logic [31:0] value_b
    );
        begin
            vector_a[index*32 +: 32] = value_a;
            vector_b[index*32 +: 32] = value_b;
        end
    endtask

    task automatic run_dot_product (
        input logic [31:0] expected,
        input string description
    );
        integer launch_cycle;
        begin
            @(negedge clk);
            start = 1;
            launch_cycle = cycle;
            $display("START cycle %0d: %s", launch_cycle, description);
            @(posedge clk);
            #1;
            start = 0;
            if (!valid)
                $fatal(1, "%s: valid was not asserted", description);
            if (result !== expected)
                $fatal(1, "%s: expected %08h, got %08h",
                       description, expected, result);
            $display("DONE  cycle %0d: %s [FP32 bits: %08h]",
                     cycle, description, result);
            if (cycle != (launch_cycle + 1))
                $fatal(1, "%s: expected one-cycle latency", description);

            // valid is a pulse and must fall when there is no new command.
            @(posedge clk);
            #1;
            if (valid)
                $fatal(1, "%s: valid did not return low", description);
        end
    endtask

    integer i;
    initial begin
        rst = 1;
        start = 0;
        vector_a = 0;
        vector_b = 0;
        cycle = 0;

        repeat (2) @(negedge clk);
        rst = 0;

        // All 16 lanes are active simultaneously: 16 * (1.0 * 1.0) = 16.0.
        for (i = 0; i < 16; i = i + 1)
            set_element(i, 32'h3f80_0000, 32'h3f80_0000);
        run_dot_product(32'h4180_0000,
                        "16 parallel products: sum(1.0 * 1.0) = 16.0");

        // Recreate the earlier three-element example; unused lanes are zero.
        vector_a = 0;
        vector_b = 0;
        set_element(0, 32'h3fc0_0000, 32'h4080_0000); //  1.5  *  4
        set_element(1, 32'hc010_0000, 32'h4040_0000); // -2.25 *  3
        set_element(2, 32'h3f00_0000, 32'hc100_0000); //  0.5  * -8
        run_dot_product(32'hc098_0000,
                        "1.5*4.0 + -2.25*3.0 + 0.5*-8.0 = -4.75");

        // Fractional workload using every lane: 16 * (0.5 * 0.25) = 2.0.
        for (i = 0; i < 16; i = i + 1)
            set_element(i, 32'h3f00_0000, 32'h3e80_0000);
        run_dot_product(32'h4000_0000,
                        "16 parallel products: sum(0.5 * 0.25) = 2.0");

        $display("ALL PARALLEL FP32 DOT-PRODUCT TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
