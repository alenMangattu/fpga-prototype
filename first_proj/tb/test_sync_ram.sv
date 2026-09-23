module test_sync_ram;
    logic clk = 0;
    always #5 clk = ~clk;

    logic cs = 0;
    logic we = 0;
    logic [3:0] addr = 0;
    logic [7:0] write_data = 0;
    logic [7:0] read_data;

    sync_ram #(.ADDR_WIDTH(4), .DATA_WIDTH(8)) dut (.*);

    initial begin
        @(negedge clk);
        cs = 1;
        we = 1;
        addr = 4'd3;
        write_data = 8'hA5;
        @(posedge clk);
        #1;
        if (read_data !== 8'hxx) $fatal(1, "write changed read_data");

        @(negedge clk);
        we = 0;
        @(posedge clk);
        #1;
        if (read_data !== 8'hA5) $fatal(1, "read failed");

        @(negedge clk);
        cs = 0;
        addr = 4'd4;
        @(posedge clk);
        #1;
        if (read_data !== 8'hA5) $fatal(1, "read_data did not hold");

        $display("PASS: SRAM write, synchronous read, and hold");
        $finish;
    end
endmodule
