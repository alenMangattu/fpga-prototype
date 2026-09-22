`timescale 1ns/1ps
module tb_q8_0_dot_pipeline;
    reg clk=0, rst=1, input_valid=0;
    reg [255:0] weights=0, activations=0;
    reg [15:0] weight_scale=0, activation_scale=0;
    reg [15:0] input_tag=0;
    wire output_valid;
    wire signed [31:0] integer_dot;
    wire [15:0] output_weight_scale, output_activation_scale, output_tag;
    integer expected [0:255];
    reg [5:0] valid_history=0;
    integer cycle, lane, a, b, sent=0, received=0, total;
    q8_0_dot_pipeline #(.TAG_BITS(16)) dut(.*);
    always #5 clk=~clk;
    initial begin
        repeat (2) @(negedge clk);
        rst=0;
        for (cycle=0; cycle<256; cycle=cycle+1) begin
            input_valid=(cycle<248 && cycle%7 != 3);
            input_tag=16'(sent);
            weight_scale=16'(sent+37);
            activation_scale=16'(sent+73);
            total=0;
            for (lane=0; lane<32; lane=lane+1) begin
                if (sent==0) begin a=-128; b=-128; end
                else if (sent==1) begin a=-128; b=127; end
                else if (sent==2) begin a=127; b=127; end
                else begin
                    a=((cycle*31+lane*19)%256)-128;
                    b=((cycle*13+lane*47)%256)-128;
                end
                weights[8*lane +: 8]=8'(a);
                activations[8*lane +: 8]=8'(b);
                total=total+a*b;
            end
            if (input_valid) begin expected[sent]=total; sent=sent+1; end
            valid_history={valid_history[4:0],input_valid};
            @(posedge clk); #1;
            if (output_valid !== valid_history[5]) $fatal(1,"valid latency/bubble mismatch");
            if (output_valid) begin
                if (output_tag !== received || integer_dot !== expected[received])
                    $fatal(1,"dot/tag mismatch at %0d",received);
                if (output_weight_scale !== 16'(received+37) || output_activation_scale !== 16'(received+73))
                    $fatal(1,"scale metadata mismatch");
                received=received+1;
            end
            @(negedge clk);
        end
        if (received != sent) $fatal(1,"lost outputs");
        $display("Q8 PIPELINE PASSED: %0d signed dots, extrema, bubbles, scales and tags",received);
        $finish;
    end
endmodule
