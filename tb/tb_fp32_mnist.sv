`timescale 1ns/1ps
`default_nettype none

module tb_fp32_mnist;
    localparam integer INPUT_SIZE = 784;
    localparam integer HIDDEN_SIZE = 100;
    localparam integer OUTPUT_SIZE = 10;

    logic clk;
    logic rst;
    logic start;
    logic [INPUT_SIZE*32-1:0] pixels;
    logic [HIDDEN_SIZE*INPUT_SIZE*32-1:0] weights_1;
    logic [HIDDEN_SIZE*32-1:0] bias_1;
    logic [OUTPUT_SIZE*HIDDEN_SIZE*32-1:0] weights_2;
    logic [OUTPUT_SIZE*32-1:0] bias_2;
    logic [HIDDEN_SIZE*32-1:0] hidden_activations;
    logic [OUTPUT_SIZE*32-1:0] scores;
    logic [$clog2(OUTPUT_SIZE)-1:0] predicted_class;
    logic busy;
    logic done;
    logic [31:0] inference_cycles;
    integer cycle;
    integer i;

    fp32_mnist dut (
        .clk(clk), .rst(rst), .start(start), .pixels(pixels),
        .weights_1(weights_1), .bias_1(bias_1),
        .weights_2(weights_2), .bias_2(bias_2),
        .hidden_activations(hidden_activations), .scores(scores),
        .predicted_class(predicted_class),
        .busy(busy), .done(done), .inference_cycles(inference_cycles)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk)
        cycle <= cycle + 1;

    task automatic set_pixel(input integer index, input logic [31:0] value);
        pixels[index*32 +: 32] = value;
    endtask

    task automatic set_w1(
        input integer row, input integer column, input logic [31:0] value
    );
        weights_1[(row*INPUT_SIZE + column)*32 +: 32] = value;
    endtask

    task automatic set_b1(input integer index, input logic [31:0] value);
        bias_1[index*32 +: 32] = value;
    endtask

    task automatic set_w2(
        input integer row, input integer column, input logic [31:0] value
    );
        weights_2[(row*HIDDEN_SIZE + column)*32 +: 32] = value;
    endtask

    task automatic set_b2(input integer index, input logic [31:0] value);
        bias_2[index*32 +: 32] = value;
    endtask

    task automatic check_score(
        input integer index,
        input logic [31:0] expected,
        input string decimal_value
    );
        logic [31:0] actual;
        begin
            actual = scores[index*32 +: 32];
            if (actual !== expected)
                $fatal(1, "score[%0d]: expected %08h, got %08h",
                       index, expected, actual);
            $display("  score[%0d] = %s [FP32 bits: %08h]",
                     index, decimal_value, actual);
        end
    endtask

    initial begin
        rst = 1;
        start = 0;
        pixels = 0;
        weights_1 = 0;
        bias_1 = 0;
        weights_2 = 0;
        bias_2 = 0;
        cycle = 0;

        // Sparse, deterministic 784-100-10 test network. Hidden neuron i
        // reads pixel i. Pixel 0 is negative, demonstrating ReLU.
        for (i = 0; i < HIDDEN_SIZE; i = i + 1)
            set_w1(i, i, 32'h3f80_0000); // 1.0

        set_pixel(0, 32'hc000_0000); // -2.0 -> ReLU -> 0.0
        for (i = 1; i < 10; i = i + 1)
            set_pixel(i, 32'h3f80_0000); // 1.0

        // Hidden neuron 99 receives a positive bias of 0.25.
        set_b1(99, 32'h3e80_0000);

        // Output i reads hidden neuron i. Output 9 also adds 2*hidden[99].
        for (i = 0; i < OUTPUT_SIZE; i = i + 1)
            set_w2(i, i, 32'h3f80_0000);
        set_w2(9, 99, 32'h4000_0000); // 2.0
        set_b2(0, 32'h3f00_0000);     // 0.5

        repeat (2) @(negedge clk);
        rst = 0;
        @(negedge clk);
        start = 1;
        $display("START: MNIST topology 784 -> 100 ReLU -> 10");
        @(negedge clk);
        start = 0;

        fork
            begin
                wait (done);
            end
            begin
                repeat (1000) @(posedge clk);
                $fatal(1, "MNIST inference timed out");
            end
        join_any
        disable fork;
        #1;

        if (busy)
            $fatal(1, "busy remained high when done asserted");
        if (hidden_activations[0*32 +: 32] !== 32'h0000_0000)
            $fatal(1, "ReLU did not clamp hidden[0] to zero");
        for (i = 1; i < 10; i = i + 1)
            if (hidden_activations[i*32 +: 32] !== 32'h3f80_0000)
                $fatal(1, "hidden[%0d] should equal 1.0", i);
        for (i = 10; i < 99; i = i + 1)
            if (hidden_activations[i*32 +: 32] !== 32'h0000_0000)
                $fatal(1, "hidden[%0d] should equal 0.0", i);
        if (hidden_activations[99*32 +: 32] !== 32'h3e80_0000)
            $fatal(1, "hidden[99] should equal its 0.25 bias");

        $display("DONE: inference completed in %0d processing clocks",
                 inference_cycles);
        check_score(0, 32'h3f00_0000, "0.5");
        for (i = 1; i < 9; i = i + 1)
            check_score(i, 32'h3f80_0000, "1.0");
        check_score(9, 32'h3fc0_0000, "1.5");

        if (predicted_class !== 4'd9)
            $fatal(1, "argmax expected class 9, got %0d", predicted_class);
        $display("PREDICTED CLASS = %0d (largest score is 1.5)",
                 predicted_class);
        $display("ALL FP32 MNIST-SIZED DNN TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
