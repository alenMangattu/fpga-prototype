`timescale 1ns/1ps
module test_matmul5_fp32;
    logic clk = 0;
    always #5 clk = ~clk;

    logic rst_n = 0;
    logic wr_valid = 0, wr_ready, wr_bank = 0;
    logic [4:0] wr_addr = 0;
    logic [31:0] wr_data = 0;
    logic cmd_valid = 0, cmd_ready;
    logic result_valid, result_ready = 0;
    logic [4:0] result_addr;
    logic [31:0] result_data;

    matmul5_fp32 dut (.*);

    function automatic logic [31:0] pattern(input int i);
        case (i % 4)
            0: pattern = 32'h3f800000; // +1.0
            1: pattern = 32'h40000000; // +2.0
            2: pattern = 32'hbf800000; // -1.0
            3: pattern = 32'h3f000000; // +0.5
        endcase
    endfunction

    function automatic int a_value(input int i);
        a_value = ((i * 3 + 1) % 7) - 3;
    endfunction

    function automatic int b_value(input int i);
        b_value = ((i * 2 + 3) % 7) - 3;
    endfunction

    function automatic logic [31:0] float_from_int(input int n);
        int magnitude;
        int exponent;
        if (n == 0) begin
            float_from_int = 32'h00000000;
        end else begin
            magnitude = (n < 0) ? -n : n;
            exponent = 0;
            for (int bit_index = 0; bit_index < 31; bit_index++)
                if (magnitude >> bit_index) exponent = bit_index;
            float_from_int = {(n < 0), 8'(exponent + 127),
                              23'((magnitude << (23 - exponent)) & 'h7fffff)};
        end
    endfunction

    task automatic write_word(input logic bank, input int addr,
                              input logic [31:0] data);
        @(negedge clk);
        wr_bank = bank;
        wr_addr = addr;
        wr_data = data;
        wr_valid = 1;
        @(posedge clk);
        if (!wr_ready) $fatal(1, "write rejected at %0d", addr);
        @(negedge clk);
        wr_valid = 0;
    endtask

    task automatic start_run;
        @(negedge clk);
        cmd_valid = 1;
        @(posedge clk);
        if (!cmd_ready) $fatal(1, "start rejected");
        @(negedge clk);
        cmd_valid = 0;
    endtask

    task automatic check_results(input int mode);
        logic [31:0] expected;
        int expected_int;
        for (int i = 0; i < 25; i++) begin
            do @(posedge clk); while (!result_valid);
            case (mode)
                0: expected = 32'h40a00000; // 5.0
                1: expected = pattern(i);
                default: begin
                    expected_int = 0;
                    for (int j = 0; j < 5; j++)
                        expected_int += a_value((i / 5) * 5 + j)
                                        * b_value(j * 5 + (i % 5));
                    expected = float_from_int(expected_int);
                end
            endcase
            if (result_addr !== i || result_data !== expected)
                $fatal(1, "C[%0d] got addr=%0d data=%h expected=%h",
                       i, result_addr, result_data, expected);
            repeat (3) begin
                @(posedge clk);
                if (!result_valid || result_addr !== i || result_data !== expected)
                    $fatal(1, "result changed under backpressure");
            end
            @(negedge clk);
            result_ready = 1;
            @(posedge clk);
            @(negedge clk);
            result_ready = 0;
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        if (cmd_ready) $fatal(1, "start accepted before loading");

        // Ones times ones: every output is 5.0.
        for (int i = 0; i < 25; i++) begin
            write_word(0, i, 32'h3f800000);
            write_word(1, i, 32'h3f800000);
        end
        start_run();
        check_results(0);

        // Identity times patterned B: every output equals the matching B word.
        for (int i = 0; i < 25; i++) begin
            write_word(0, i, ((i / 5) == (i % 5)) ? 32'h3f800000 : 32'h00000000);
            write_word(1, i, pattern(i));
        end
        start_run();
        check_results(1);

        // Vary both rows and columns, including negative operands.
        for (int i = 0; i < 25; i++) begin
            write_word(0, i, float_from_int(a_value(i)));
            write_word(1, i, float_from_int(b_value(i)));
        end
        start_run();
        check_results(2);

        $display("PASS: three FP32 5x5 matrix products, 75 outputs, backpressure");
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "simulation timeout");
    end
endmodule
