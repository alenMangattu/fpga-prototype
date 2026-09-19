`timescale 1ns/1ps
`default_nettype none

module tb_fp32_dnn;
    localparam integer INPUT_SIZE = 4;
    localparam integer HIDDEN_SIZE = 3;
    localparam integer OUTPUT_SIZE = 2;

    logic clk;
    logic rst;
    logic start;
    logic [INPUT_SIZE*32-1:0] input_vector;
    logic [HIDDEN_SIZE*INPUT_SIZE*32-1:0] weights_1;
    logic [HIDDEN_SIZE*32-1:0] bias_1;
    logic [OUTPUT_SIZE*HIDDEN_SIZE*32-1:0] weights_2;
    logic [OUTPUT_SIZE*32-1:0] bias_2;
    logic [HIDDEN_SIZE*32-1:0] hidden_vector;
    logic [OUTPUT_SIZE*32-1:0] output_vector;
    logic valid;
    integer cycle;

    fp32_dnn #(
        .INPUT_SIZE(INPUT_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .input_vector(input_vector),
        .weights_1(weights_1), .bias_1(bias_1),
        .weights_2(weights_2), .bias_2(bias_2),
        .hidden_vector(hidden_vector),
        .output_vector(output_vector), .valid(valid)
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

    task automatic set_input(input integer i, input logic [31:0] value);
        input_vector[i*32 +: 32] = value;
    endtask

    task automatic set_w1(
        input integer row, input integer column, input logic [31:0] value
    );
        weights_1[(row*INPUT_SIZE + column)*32 +: 32] = value;
    endtask

    task automatic set_b1(input integer i, input logic [31:0] value);
        bias_1[i*32 +: 32] = value;
    endtask

    task automatic set_w2(
        input integer row, input integer column, input logic [31:0] value
    );
        weights_2[(row*HIDDEN_SIZE + column)*32 +: 32] = value;
    endtask

    task automatic set_b2(input integer i, input logic [31:0] value);
        bias_2[i*32 +: 32] = value;
    endtask

    task automatic check_output(
        input logic [31:0] expected_0,
        input logic [31:0] expected_1,
        input string description
    );
        begin
            if (!valid)
                $fatal(1, "%s: output was not valid", description);
            if (output_vector[0*32 +: 32] !== expected_0)
                $fatal(1, "%s: output[0] expected %08h, got %08h",
                       description, expected_0, output_vector[0*32 +: 32]);
            if (output_vector[1*32 +: 32] !== expected_1)
                $fatal(1, "%s: output[1] expected %08h, got %08h",
                       description, expected_1, output_vector[1*32 +: 32]);
            $display("DONE cycle %0d: %s", cycle, description);
            $display("  output = [%08h, %08h]",
                     output_vector[0*32 +: 32], output_vector[1*32 +: 32]);
        end
    endtask

    initial begin
        rst = 1;
        start = 0;
        input_vector = 0;
        weights_1 = 0;
        bias_1 = 0;
        weights_2 = 0;
        bias_2 = 0;
        cycle = 0;

        // W1 = [[1, 1, 1, 1], [-1, .5, 2, -1], [2, -1, 0, .5]]
        set_w1(0, 0, 32'h3f80_0000); set_w1(0, 1, 32'h3f80_0000);
        set_w1(0, 2, 32'h3f80_0000); set_w1(0, 3, 32'h3f80_0000);
        set_w1(1, 0, 32'hbf80_0000); set_w1(1, 1, 32'h3f00_0000);
        set_w1(1, 2, 32'h4000_0000); set_w1(1, 3, 32'hbf80_0000);
        set_w1(2, 0, 32'h4000_0000); set_w1(2, 1, 32'hbf80_0000);
        set_w1(2, 2, 32'h0000_0000); set_w1(2, 3, 32'h3f00_0000);

        // b1 = [0.5, 1.0, -0.5]
        set_b1(0, 32'h3f00_0000);
        set_b1(1, 32'h3f80_0000);
        set_b1(2, 32'hbf00_0000);

        // W2 = [[2, 1, -1], [-1, 3, .5]], b2 = [.25, -.5]
        set_w2(0, 0, 32'h4000_0000); set_w2(0, 1, 32'h3f80_0000);
        set_w2(0, 2, 32'hbf80_0000); set_w2(1, 0, 32'hbf80_0000);
        set_w2(1, 1, 32'h4040_0000); set_w2(1, 2, 32'h3f00_0000);
        set_b2(0, 32'h3e80_0000);
        set_b2(1, 32'hbf00_0000);

        repeat (2) @(negedge clk);
        rst = 0;

        // Inference 1: x=[1,-2,.5,3], hidden=ReLU([3,-3,5])=[3,0,5]
        // and output=[1.25,-1].
        set_input(0, 32'h3f80_0000);
        set_input(1, 32'hc000_0000);
        set_input(2, 32'h3f00_0000);
        set_input(3, 32'h4040_0000);
        @(negedge clk);
        start = 1;
        $display("START cycle %0d: x1=[1.0,-2.0,0.5,3.0]", cycle);

        // Launch inference 2 on the immediately following cycle to verify
        // one-inference-per-clock pipeline throughput. x2=[0,0,0,0].
        @(posedge clk);
        #1;
        if (valid)
            $fatal(1, "Output arrived too early");
        if ((hidden_vector[0*32 +: 32] !== 32'h4040_0000) ||
            (hidden_vector[1*32 +: 32] !== 32'h0000_0000) ||
            (hidden_vector[2*32 +: 32] !== 32'h40a0_0000))
            $fatal(1, "x1 hidden layer did not equal ReLU([3,-3,5])=[3,0,5]");
        $display("HIDDEN cycle %0d: ReLU([3.0,-3.0,5.0]) = [3.0,0.0,5.0]",
                 cycle);
        @(negedge clk);
        set_input(0, 32'h0000_0000); set_input(1, 32'h0000_0000);
        set_input(2, 32'h0000_0000); set_input(3, 32'h0000_0000);
        start = 1;
        $display("START cycle %0d: x2=[0.0,0.0,0.0,0.0]", cycle);

        @(posedge clk);
        #1;
        check_output(32'h3fa0_0000, 32'hbf80_0000,
                     "x1 output=[1.25,-1.0], latency=2 clocks");

        @(negedge clk);
        start = 0;
        @(posedge clk);
        #1;
        check_output(32'h4010_0000, 32'h4000_0000,
                     "x2 output=[2.25,2.0], one-clock throughput");

        @(posedge clk);
        #1;
        if (valid)
            $fatal(1, "valid did not return low");

        $display("ALL FP32 DNN TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
