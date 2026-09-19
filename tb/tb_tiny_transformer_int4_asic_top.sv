`timescale 1ns/1ps
`default_nettype none

module tb_tiny_transformer_int4_asic_top;
    logic clk = 0;
    logic reset_n = 0;
    logic start = 0;
    logic cfg_we = 0;
    logic [9:0] cfg_addr = 0;
    logic [31:0] cfg_wdata = 0;
    logic [31:0] cfg_rdata;
    logic cfg_ready, busy, done;
    logic [7:0] predicted_token;
    logic [31:0] cycle_count, parameter_count, parameter_bits;
    integer address;

    always #5 clk = ~clk;
    tiny_transformer_int4_asic_top dut (.*);

    task automatic write_word(input integer addr, input logic [31:0] data);
        begin
            @(negedge clk);
            cfg_addr = addr;
            cfg_wdata = data;
            cfg_we = 1;
            @(negedge clk);
            cfg_we = 0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        reset_n = 1;
        for (address = 0; address < 27; address = address + 1)
            write_word(address, 0);

        // Token 1 embedding=1, token 2 embedding=2 in dimension zero.
        write_word(0, 32'h0002_0100);
        // Identity Q/K/V/output matrices (row-major 2x2 INT4).
        write_word(2, 32'h0000_1001);
        write_word(5, 32'h0000_1001);
        write_word(8, 32'h0000_1001);
        write_word(11, 32'h0000_1001);
        // LM class 3 reads hidden dimension zero.
        write_word(22, 32'h0100_0000);
        write_word(512, 32'h0000_0201);

        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        wait (done);
        #1;

        if (parameter_count !== 70)
            $fatal(1, "parameter count mismatch: %0d", parameter_count);
        if (parameter_bits !== 784)
            $fatal(1, "parameter bits mismatch: %0d", parameter_bits);
        if (predicted_token !== 3)
            $fatal(1, "expected token 3, got %0d", predicted_token);
        $display("INT4 ASIC WRAPPER PASSED: token=%0d cycles=%0d bits=%0d",
                 predicted_token, cycle_count, parameter_bits);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
