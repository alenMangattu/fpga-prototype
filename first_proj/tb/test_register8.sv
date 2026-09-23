module test_register8;
    logic clk = 0;
    logic rst_n = 0;
    logic enable = 0;
    logic [7:0] data_in = 0;
    logic [7:0] data_out;

    register8 dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_in(data_in),
        .data_out(data_out)
    );

    initial begin
        #5 clk = 1;              // Rising edge: reset sets output to 0
        #1 $display("reset: data_out = %0d", data_out);

        clk = 0;
        rst_n = 1;
        enable = 1;
        data_in = 42;
        #5 clk = 1;              // Rising edge: register captures 42
        #1 $display("loaded: data_out = %0d", data_out);

        $finish;
    end
endmodule