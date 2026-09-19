`timescale 1ns/1ps
`default_nettype none

module tb_tiny_transformer_int4;
    localparam integer VOCAB_SIZE = 256;
    localparam integer SEQ_LEN = 4;
    localparam integer D_MODEL = 16;
    localparam integer D_FF = 32;

    logic clk = 0;
    logic rst = 1;
    logic start = 0;
    logic [SEQ_LEN*8-1:0] token_ids = 0;
    logic [VOCAB_SIZE*D_MODEL*4-1:0] token_embedding = 0;
    logic [SEQ_LEN*D_MODEL*4-1:0] position_embedding = 0;
    logic [D_MODEL*D_MODEL*4-1:0] weight_q = 0;
    logic [D_MODEL*D_MODEL*4-1:0] weight_k = 0;
    logic [D_MODEL*D_MODEL*4-1:0] weight_v = 0;
    logic [D_MODEL*D_MODEL*4-1:0] weight_o = 0;
    logic [D_MODEL*32-1:0] bias_q = 0;
    logic [D_MODEL*32-1:0] bias_k = 0;
    logic [D_MODEL*32-1:0] bias_v = 0;
    logic [D_MODEL*32-1:0] bias_o = 0;
    logic [D_FF*D_MODEL*4-1:0] weight_ff1 = 0;
    logic [D_FF*32-1:0] bias_ff1 = 0;
    logic [D_MODEL*D_FF*4-1:0] weight_ff2 = 0;
    logic [D_MODEL*32-1:0] bias_ff2 = 0;
    logic [VOCAB_SIZE*D_MODEL*4-1:0] weight_lm = 0;
    logic [VOCAB_SIZE*32-1:0] bias_lm = 0;
    logic [VOCAB_SIZE*32-1:0] logits;
    logic [7:0] predicted_token;
    logic [D_MODEL*8-1:0] last_hidden;
    logic busy, done;
    logic [31:0] cycle_count, parameter_count, parameter_bits;
    integer i;

    always #5 clk = ~clk;

    tiny_transformer_int4 #(
        .VOCAB_SIZE(VOCAB_SIZE), .SEQ_LEN(SEQ_LEN),
        .D_MODEL(D_MODEL), .D_FF(D_FF)
    ) dut (.*);

    task automatic set_square_weight(
        input integer matrix_number,
        input integer row,
        input integer column,
        input logic [3:0] value
    );
        case (matrix_number)
            0: weight_q[(row*D_MODEL + column)*4 +: 4] = value;
            1: weight_k[(row*D_MODEL + column)*4 +: 4] = value;
            2: weight_v[(row*D_MODEL + column)*4 +: 4] = value;
            3: weight_o[(row*D_MODEL + column)*4 +: 4] = value;
            default: $fatal(1, "invalid matrix number");
        endcase
    endtask

    initial begin
        token_ids[0*8 +: 8] = 8'd1;
        token_ids[1*8 +: 8] = 8'd2;
        token_ids[2*8 +: 8] = 8'd3;
        token_ids[3*8 +: 8] = 8'd4;
        token_embedding[(1*D_MODEL + 0)*4 +: 4] = 4'sd1;
        token_embedding[(2*D_MODEL + 0)*4 +: 4] = 4'sd2;
        token_embedding[(3*D_MODEL + 0)*4 +: 4] = 4'sd3;
        token_embedding[(4*D_MODEL + 0)*4 +: 4] = 4'sd4;

        for (i = 0; i < D_MODEL; i = i + 1) begin
            set_square_weight(0, i, i, 4'sd1);
            set_square_weight(1, i, i, 4'sd1);
            set_square_weight(2, i, i, 4'sd1);
            set_square_weight(3, i, i, 4'sd1);
        end
        weight_lm[(9*D_MODEL + 0)*4 +: 4] = 4'sd1;

        repeat (2) @(negedge clk);
        rst = 0;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        fork
            wait (done);
            begin
                repeat (700) @(posedge clk);
                $fatal(1, "INT4 transformer timed out");
            end
        join_any
        disable fork;
        #1;

        if (parameter_count !== 32'd10672)
            $fatal(1, "parameter count mismatch: %0d", parameter_count);
        if (parameter_bits !== 32'd52992)
            $fatal(1, "parameter bit count mismatch: %0d", parameter_bits);
        if ($signed(last_hidden[0 +: 8]) !== 8'sd8)
            $fatal(1, "last hidden[0] expected 8, got %0d",
                   $signed(last_hidden[0 +: 8]));
        if ($signed(logits[9*32 +: 32]) !== 32'sd8)
            $fatal(1, "token-9 logit expected 8, got %0d",
                   $signed(logits[9*32 +: 32]));
        if (predicted_token !== 8'd9)
            $fatal(1, "expected token 9, got %0d", predicted_token);

        $display("INT4 TRANSFORMER PASSED: token=%0d cycles=%0d bits=%0d",
                 predicted_token, cycle_count, parameter_bits);
        $finish;
    end
endmodule

`default_nettype wire
