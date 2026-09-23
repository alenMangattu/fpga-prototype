module test_fake_dram;
    localparam int LATENCY = 3;
    logic clk = 0;
    always #5 clk = ~clk;

    logic rst_n = 0;
    logic req_valid = 0;
    logic req_ready;
    logic req_write = 0;
    logic [3:0] req_addr = 0;
    logic [7:0] req_wdata = 0;
    logic resp_valid;
    logic [7:0] resp_rdata;

    fake_dram #(.ADDR_WIDTH(4), .DATA_WIDTH(8), .LATENCY(LATENCY)) dut (.*);

    task automatic request(input logic write_op, input logic [3:0] addr,
                           input logic [7:0] data);
        @(negedge clk);
        if (!req_ready) $fatal(1, "DRAM not ready for request");
        req_write = write_op;
        req_addr = addr;
        req_wdata = data;
        req_valid = 1;
        @(posedge clk);
        #1;
        if (req_ready || resp_valid) $fatal(1, "request response too early");
        @(negedge clk);
        req_valid = 0;
        for (int i = 1; i < LATENCY; i++) begin
            @(posedge clk);
            #1;
            if (resp_valid) $fatal(1, "response arrived too early");
        end
        @(posedge clk);
        #1;
        if (!resp_valid || !req_ready) $fatal(1, "response missing");
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1;
        request(1, 4'd3, 8'hA5); // write A5 to address 3
        request(0, 4'd3, 8'h00); // read address 3
        if (resp_rdata !== 8'hA5)
            $fatal(1, "read %h instead of A5", resp_rdata);
        $display("PASS: delayed write and read, latency=%0d clocks", LATENCY);
        $finish;
    end
endmodule
