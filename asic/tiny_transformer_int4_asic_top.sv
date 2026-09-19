`timescale 1ns/1ps
`default_nettype none

// Packed W4A8 ASIC wrapper for the smallest transformer proof core.
// INT4 arrays are packed eight values per 32-bit configuration word;
// INT32 biases occupy one word each.
module tiny_transformer_int4_asic_top (
    input  logic        clk,
    input  logic        reset_n,
    input  logic        start,
    input  logic        cfg_we,
    input  logic [9:0]  cfg_addr,
    input  logic [31:0] cfg_wdata,
    output logic [31:0] cfg_rdata,
    output logic        cfg_ready,
    output logic        busy,
    output logic        done,
    output logic [7:0]  predicted_token,
    output logic [31:0] cycle_count,
    output logic [31:0] parameter_count,
    output logic [31:0] parameter_bits
);
    localparam integer VOCAB_SIZE = 4;
    localparam integer SEQ_LEN = 2;
    localparam integer D_MODEL = 2;
    localparam integer D_FF = 4;
    localparam integer CONFIG_WORDS = 27;
    localparam logic [9:0] TOKEN_IDS_ADDRESS = 10'd512;

    localparam integer TOKEN_EMBED_WORD = 0;
    localparam integer POSITION_WORD = 1;
    localparam integer WQ_WORD = 2;
    localparam integer BQ_WORD = 3;
    localparam integer WK_WORD = 5;
    localparam integer BK_WORD = 6;
    localparam integer WV_WORD = 8;
    localparam integer BV_WORD = 9;
    localparam integer WO_WORD = 11;
    localparam integer BO_WORD = 12;
    localparam integer WFF1_WORD = 14;
    localparam integer BFF1_WORD = 15;
    localparam integer WFF2_WORD = 19;
    localparam integer BFF2_WORD = 20;
    localparam integer WLM_WORD = 22;
    localparam integer BLM_WORD = 23;

    logic rst;
    logic [31:0] parameter_memory [0:CONFIG_WORDS-1];
    logic [SEQ_LEN*8-1:0] token_ids;
    logic [VOCAB_SIZE*D_MODEL*4-1:0] token_embedding;
    logic [SEQ_LEN*D_MODEL*4-1:0] position_embedding;
    logic [D_MODEL*D_MODEL*4-1:0] weight_q, weight_k, weight_v, weight_o;
    logic [D_MODEL*32-1:0] bias_q, bias_k, bias_v, bias_o;
    logic [D_FF*D_MODEL*4-1:0] weight_ff1;
    logic [D_FF*32-1:0] bias_ff1;
    logic [D_MODEL*D_FF*4-1:0] weight_ff2;
    logic [D_MODEL*32-1:0] bias_ff2;
    logic [VOCAB_SIZE*D_MODEL*4-1:0] weight_lm;
    logic [VOCAB_SIZE*32-1:0] bias_lm;
    logic [VOCAB_SIZE*32-1:0] unused_logits;
    logic [D_MODEL*8-1:0] unused_last_hidden;

    assign rst = !reset_n;
    assign cfg_ready = !busy;
    assign token_embedding = parameter_memory[TOKEN_EMBED_WORD];
    assign position_embedding = parameter_memory[POSITION_WORD][15:0];
    assign weight_q = parameter_memory[WQ_WORD][15:0];
    assign weight_k = parameter_memory[WK_WORD][15:0];
    assign weight_v = parameter_memory[WV_WORD][15:0];
    assign weight_o = parameter_memory[WO_WORD][15:0];
    assign weight_ff1 = parameter_memory[WFF1_WORD];
    assign weight_ff2 = parameter_memory[WFF2_WORD];
    assign weight_lm = parameter_memory[WLM_WORD];

    genvar i;
    generate
        for (i = 0; i < D_MODEL; i = i + 1) begin : model_biases
            assign bias_q[i*32 +: 32] = parameter_memory[BQ_WORD+i];
            assign bias_k[i*32 +: 32] = parameter_memory[BK_WORD+i];
            assign bias_v[i*32 +: 32] = parameter_memory[BV_WORD+i];
            assign bias_o[i*32 +: 32] = parameter_memory[BO_WORD+i];
            assign bias_ff2[i*32 +: 32] = parameter_memory[BFF2_WORD+i];
        end
        for (i = 0; i < D_FF; i = i + 1)
            assign bias_ff1[i*32 +: 32] = parameter_memory[BFF1_WORD+i];
        for (i = 0; i < VOCAB_SIZE; i = i + 1)
            assign bias_lm[i*32 +: 32] = parameter_memory[BLM_WORD+i];
    endgenerate

    always @* begin
        if (cfg_addr < CONFIG_WORDS)
            cfg_rdata = parameter_memory[cfg_addr];
        else if (cfg_addr == TOKEN_IDS_ADDRESS)
            cfg_rdata = {{(32-SEQ_LEN*8){1'b0}}, token_ids};
        else
            cfg_rdata = 32'd0;
    end

    always_ff @(posedge clk) begin
        if (!reset_n)
            token_ids <= 0;
        else if (cfg_we && cfg_ready) begin
            if (cfg_addr < CONFIG_WORDS)
                parameter_memory[cfg_addr] <= cfg_wdata;
            else if (cfg_addr == TOKEN_IDS_ADDRESS)
                token_ids <= cfg_wdata[SEQ_LEN*8-1:0];
        end
    end

    tiny_transformer_int4 #(
        .VOCAB_SIZE(VOCAB_SIZE), .SEQ_LEN(SEQ_LEN),
        .D_MODEL(D_MODEL), .D_FF(D_FF)
    ) core (
        .clk(clk), .rst(rst), .start(start && !cfg_we),
        .token_ids(token_ids), .token_embedding(token_embedding),
        .position_embedding(position_embedding),
        .weight_q(weight_q), .bias_q(bias_q),
        .weight_k(weight_k), .bias_k(bias_k),
        .weight_v(weight_v), .bias_v(bias_v),
        .weight_o(weight_o), .bias_o(bias_o),
        .weight_ff1(weight_ff1), .bias_ff1(bias_ff1),
        .weight_ff2(weight_ff2), .bias_ff2(bias_ff2),
        .weight_lm(weight_lm), .bias_lm(bias_lm),
        .logits(unused_logits), .predicted_token(predicted_token),
        .last_hidden(unused_last_hidden), .busy(busy), .done(done),
        .cycle_count(cycle_count), .parameter_count(parameter_count),
        .parameter_bits(parameter_bits)
    );
endmodule

`default_nettype wire
