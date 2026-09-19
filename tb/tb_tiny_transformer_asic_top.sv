`timescale 1ns/1ps
`default_nettype none

module tb_tiny_transformer_asic_top;
    logic clk = 0;
    logic reset_n = 0;
    logic start = 0;
    logic cfg_we = 0;
    logic [9:0] cfg_addr = 0;
    logic [31:0] cfg_wdata = 0;
    logic [31:0] cfg_rdata;
    logic cfg_ready, busy, done;
    logic [7:0] predicted_token;
    logic [31:0] cycle_count, parameter_count;
    integer address;

    always #5 clk = ~clk;
    tiny_transformer_asic_top dut (.*);

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
        for (address = 0; address < 70; address = address + 1)
            write_word(address, 32'h0000_0000);
        write_word(512, 32'h0000_0102);

        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        wait (done);
        if (parameter_count !== 70)
            $fatal(1, "parameter count mismatch: %0d", parameter_count);
        if (predicted_token !== 0)
            $fatal(1, "zero model should select token zero, got %0d", predicted_token);
        $display("ASIC WRAPPER PASSED: token=%0d cycles=%0d params=%0d",
                 predicted_token, cycle_count, parameter_count);
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
