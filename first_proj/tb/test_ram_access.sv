module test_ram_access;
    logic clk = 0;
    always #5 clk = ~clk;

    logic write_en = 0;
    logic [3:0] write_addr = 0;
    logic [7:0] write_data = 0;
    logic read_en = 0;
    logic [3:0] read_addr = 0;
    logic [7:0] read_data;

    ram_access #(.ADDR_WIDTH(4), .DATA_WIDTH(8)) dut (.*);

    initial begin
        // Instruction-like request: store 0xA5 at address 3.
        @(negedge clk);
        write_addr = 4'd3;
        write_data = 8'hA5;
        write_en = 1;
        @(negedge clk);
        write_en = 0;

        // Read address 3 on the next rising edge.
        read_addr = 4'd3;
        read_en = 1;
        @(posedge clk);
        #1;
        if (read_data !== 8'hA5) $fatal(1, "RAM returned %h, expected A5", read_data);
        $display("PASS: wrote A5 to address 3 and read A5 back");
        $finish;
    end
endmodule
