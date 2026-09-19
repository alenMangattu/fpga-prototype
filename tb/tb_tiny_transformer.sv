`timescale 1ns/1ps
`default_nettype none

module tb_tiny_transformer;
    localparam integer VOCAB_SIZE = 256;
    localparam integer SEQ_LEN = 4;
    localparam integer D_MODEL = 16;
    localparam integer D_FF = 32;

    logic clk, rst, start;
    logic [SEQ_LEN*8-1:0] token_ids;
    logic [VOCAB_SIZE*D_MODEL*32-1:0] token_embedding;
    logic [SEQ_LEN*D_MODEL*32-1:0] position_embedding;
    logic [D_MODEL*D_MODEL*32-1:0] weight_q, weight_k, weight_v, weight_o;
    logic [D_MODEL*32-1:0] bias_q, bias_k, bias_v, bias_o;
    logic [D_FF*D_MODEL*32-1:0] weight_ff1;
    logic [D_FF*32-1:0] bias_ff1;
    logic [D_MODEL*D_FF*32-1:0] weight_ff2;
    logic [D_MODEL*32-1:0] bias_ff2;
    logic [VOCAB_SIZE*D_MODEL*32-1:0] weight_lm;
    logic [VOCAB_SIZE*32-1:0] bias_lm;
    logic [VOCAB_SIZE*32-1:0] logits;
    logic [7:0] predicted_token;
    logic [D_MODEL*32-1:0] last_hidden;
    logic busy, done;
    logic [31:0] cycle_count, parameter_count;
    integer i;

    tiny_transformer dut (
        .clk(clk), .rst(rst), .start(start), .token_ids(token_ids),
        .token_embedding(token_embedding),
        .position_embedding(position_embedding),
        .weight_q(weight_q), .bias_q(bias_q),
        .weight_k(weight_k), .bias_k(bias_k),
        .weight_v(weight_v), .bias_v(bias_v),
        .weight_o(weight_o), .bias_o(bias_o),
        .weight_ff1(weight_ff1), .bias_ff1(bias_ff1),
        .weight_ff2(weight_ff2), .bias_ff2(bias_ff2),
        .weight_lm(weight_lm), .bias_lm(bias_lm),
        .logits(logits), .predicted_token(predicted_token),
        .last_hidden(last_hidden), .busy(busy), .done(done),
        .cycle_count(cycle_count), .parameter_count(parameter_count)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    task automatic set_token(input integer position, input logic [7:0] id);
        token_ids[position*8 +: 8] = id;
    endtask

    task automatic set_embedding(
        input integer token, input integer dimension, input logic [31:0] value
    );
        token_embedding[(token*D_MODEL + dimension)*32 +: 32] = value;
    endtask

    task automatic set_square_weight(
        input integer matrix_number,
        input integer row,
        input integer column,
        input logic [31:0] value
    );
        begin
            case (matrix_number)
                0: weight_q[(row*D_MODEL + column)*32 +: 32] = value;
                1: weight_k[(row*D_MODEL + column)*32 +: 32] = value;
                2: weight_v[(row*D_MODEL + column)*32 +: 32] = value;
                3: weight_o[(row*D_MODEL + column)*32 +: 32] = value;
                default: $fatal(1, "invalid matrix number");
            endcase
        end
    endtask

    task automatic set_lm_weight(
        input integer token, input integer dimension, input logic [31:0] value
    );
        weight_lm[(token*D_MODEL + dimension)*32 +: 32] = value;
    endtask

    initial begin
        rst = 1;
        start = 0;
        token_ids = 0;
        token_embedding = 0;
        position_embedding = 0;
        weight_q = 0; bias_q = 0;
        weight_k = 0; bias_k = 0;
        weight_v = 0; bias_v = 0;
        weight_o = 0; bias_o = 0;
        weight_ff1 = 0; bias_ff1 = 0;
        weight_ff2 = 0; bias_ff2 = 0;
        weight_lm = 0; bias_lm = 0;

        // Four input tokens whose first embedding components are 1,2,3,4.
        set_token(0, 8'd1); set_embedding(1, 0, 32'h3f80_0000);
        set_token(1, 8'd2); set_embedding(2, 0, 32'h4000_0000);
        set_token(2, 8'd3); set_embedding(3, 0, 32'h4040_0000);
        set_token(3, 8'd4); set_embedding(4, 0, 32'h4080_0000);

        // Identity Q/K/V/output projections. Since all queries are positive,
        // hardmax attention selects token 4, whose value is 4.0.
        for (i = 0; i < D_MODEL; i = i + 1) begin
            set_square_weight(0, i, i, 32'h3f80_0000);
            set_square_weight(1, i, i, 32'h3f80_0000);
            set_square_weight(2, i, i, 32'h3f80_0000);
            set_square_weight(3, i, i, 32'h3f80_0000);
        end

        // The zero FFN leaves the residual unchanged. Token 4 therefore ends
        // as 4.0 + attention_value(4.0) = 8.0. LM class 9 reads that value.
        set_lm_weight(9, 0, 32'h3f80_0000);

        repeat (2) @(negedge clk);
        rst = 0;
        @(negedge clk);
        start = 1;
        $display("START tiny transformer: tokens=[1,2,3,4]");
        @(negedge clk);
        start = 0;

        fork
            begin
                wait (done);
            end
            begin
                repeat (700) @(posedge clk);
                $fatal(1, "transformer inference timed out");
            end
        join_any
        disable fork;
        #1;

        if (parameter_count !== 32'd10672)
            $fatal(1, "expected 10672 parameters, got %0d", parameter_count);
        if (last_hidden[0 +: 32] !== 32'h4100_0000)
            $fatal(1, "last hidden[0] expected 8.0, got %08h",
                   last_hidden[0 +: 32]);
        for (i = 1; i < D_MODEL; i = i + 1)
            if (last_hidden[i*32 +: 32] !== 32'h0000_0000)
                $fatal(1, "last hidden[%0d] expected zero", i);
        if (logits[9*32 +: 32] !== 32'h4100_0000)
            $fatal(1, "token-9 logit expected 8.0, got %08h",
                   logits[9*32 +: 32]);
        for (i = 0; i < VOCAB_SIZE; i = i + 1)
            if ((i != 9) && (logits[i*32 +: 32] !== 32'h0000_0000))
                $fatal(1, "logit[%0d] expected zero, got %08h",
                       i, logits[i*32 +: 32]);
        if (predicted_token !== 8'd9)
            $fatal(1, "expected predicted token 9, got %0d", predicted_token);
        if (busy)
            $fatal(1, "busy remained high when done asserted");

        $display("DONE in %0d controller clocks", cycle_count);
        $display("PARAMETERS = %0d FP32 values", parameter_count);
        $display("LAST HIDDEN[0] = 8.0 [FP32 bits: %08h]",
                 last_hidden[0 +: 32]);
        $display("PREDICTED TOKEN = %0d", predicted_token);
        $display("ALL TINY TRANSFORMER TESTS PASSED");
        $finish;
    end
endmodule

`default_nettype wire
