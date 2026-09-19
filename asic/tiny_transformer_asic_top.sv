`timescale 1ns/1ps
`default_nettype none

// Pad-agnostic ASIC macro wrapper for the smallest tiny_transformer instance.
// Parameters are loaded through a 32-bit configuration port before inference,
// avoiding an impossible one-pin-per-weight top-level interface.
module tiny_transformer_asic_top (
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
    output logic [31:0] parameter_count
);
    localparam integer VOCAB_SIZE = 4;
    localparam integer SEQ_LEN = 2;
    localparam integer D_MODEL = 2;
    localparam integer D_FF = 4;

    localparam integer TOKEN_EMBED_OFFSET = 0;
    localparam integer TOKEN_EMBED_WORDS = VOCAB_SIZE*D_MODEL; // 8
    localparam integer POSITION_OFFSET = TOKEN_EMBED_OFFSET + TOKEN_EMBED_WORDS;
    localparam integer POSITION_WORDS = SEQ_LEN*D_MODEL;       // 4
    localparam integer WQ_OFFSET = POSITION_OFFSET + POSITION_WORDS;
    localparam integer WQ_WORDS = D_MODEL*D_MODEL;             // 4
    localparam integer BQ_OFFSET = WQ_OFFSET + WQ_WORDS;
    localparam integer WK_OFFSET = BQ_OFFSET + D_MODEL;
    localparam integer BK_OFFSET = WK_OFFSET + WQ_WORDS;
    localparam integer WV_OFFSET = BK_OFFSET + D_MODEL;
    localparam integer BV_OFFSET = WV_OFFSET + WQ_WORDS;
    localparam integer WO_OFFSET = BV_OFFSET + D_MODEL;
    localparam integer BO_OFFSET = WO_OFFSET + WQ_WORDS;
    localparam integer WFF1_OFFSET = BO_OFFSET + D_MODEL;
    localparam integer WFF1_WORDS = D_FF*D_MODEL;              // 8
    localparam integer BFF1_OFFSET = WFF1_OFFSET + WFF1_WORDS;
    localparam integer WFF2_OFFSET = BFF1_OFFSET + D_FF;
    localparam integer WFF2_WORDS = D_MODEL*D_FF;              // 8
    localparam integer BFF2_OFFSET = WFF2_OFFSET + WFF2_WORDS;
    localparam integer WLM_OFFSET = BFF2_OFFSET + D_MODEL;
    localparam integer WLM_WORDS = VOCAB_SIZE*D_MODEL;         // 8
    localparam integer BLM_OFFSET = WLM_OFFSET + WLM_WORDS;
    localparam integer PARAMETER_WORDS = BLM_OFFSET + VOCAB_SIZE; // 70
    localparam logic [9:0] TOKEN_IDS_ADDRESS = 10'd512;

    logic rst;
    logic [31:0] parameter_memory [0:PARAMETER_WORDS-1];
    logic [SEQ_LEN*8-1:0] token_ids;
    logic [TOKEN_EMBED_WORDS*32-1:0] token_embedding;
    logic [POSITION_WORDS*32-1:0] position_embedding;
    logic [WQ_WORDS*32-1:0] weight_q, weight_k, weight_v, weight_o;
    logic [D_MODEL*32-1:0] bias_q, bias_k, bias_v, bias_o;
    logic [WFF1_WORDS*32-1:0] weight_ff1;
    logic [D_FF*32-1:0] bias_ff1;
    logic [WFF2_WORDS*32-1:0] weight_ff2;
    logic [D_MODEL*32-1:0] bias_ff2;
    logic [WLM_WORDS*32-1:0] weight_lm;
    logic [VOCAB_SIZE*32-1:0] bias_lm;
    logic [VOCAB_SIZE*32-1:0] unused_logits;
    logic [D_MODEL*32-1:0] unused_last_hidden;
    assign rst = !reset_n;
    assign cfg_ready = !busy;

    always @* begin
        if (cfg_addr < PARAMETER_WORDS)
            cfg_rdata = parameter_memory[cfg_addr];
        else if (cfg_addr == TOKEN_IDS_ADDRESS)
            cfg_rdata = {{(32-SEQ_LEN*8){1'b0}}, token_ids};
        else
            cfg_rdata = 32'd0;
    end

    always_ff @(posedge clk) begin
        if (!reset_n) begin
            token_ids <= '0;
        end else if (cfg_we && cfg_ready) begin
            if (cfg_addr < PARAMETER_WORDS)
                parameter_memory[cfg_addr] <= cfg_wdata;
            else if (cfg_addr == TOKEN_IDS_ADDRESS)
                token_ids <= cfg_wdata[SEQ_LEN*8-1:0];
        end
    end

    genvar i;
    generate
        for (i = 0; i < TOKEN_EMBED_WORDS; i = i + 1)
            assign token_embedding[i*32 +: 32] = parameter_memory[TOKEN_EMBED_OFFSET+i];
        for (i = 0; i < POSITION_WORDS; i = i + 1)
            assign position_embedding[i*32 +: 32] = parameter_memory[POSITION_OFFSET+i];
        for (i = 0; i < WQ_WORDS; i = i + 1) begin
            assign weight_q[i*32 +: 32] = parameter_memory[WQ_OFFSET+i];
            assign weight_k[i*32 +: 32] = parameter_memory[WK_OFFSET+i];
            assign weight_v[i*32 +: 32] = parameter_memory[WV_OFFSET+i];
            assign weight_o[i*32 +: 32] = parameter_memory[WO_OFFSET+i];
        end
        for (i = 0; i < D_MODEL; i = i + 1) begin
            assign bias_q[i*32 +: 32] = parameter_memory[BQ_OFFSET+i];
            assign bias_k[i*32 +: 32] = parameter_memory[BK_OFFSET+i];
            assign bias_v[i*32 +: 32] = parameter_memory[BV_OFFSET+i];
            assign bias_o[i*32 +: 32] = parameter_memory[BO_OFFSET+i];
            assign bias_ff2[i*32 +: 32] = parameter_memory[BFF2_OFFSET+i];
        end
        for (i = 0; i < WFF1_WORDS; i = i + 1)
            assign weight_ff1[i*32 +: 32] = parameter_memory[WFF1_OFFSET+i];
        for (i = 0; i < D_FF; i = i + 1)
            assign bias_ff1[i*32 +: 32] = parameter_memory[BFF1_OFFSET+i];
        for (i = 0; i < WFF2_WORDS; i = i + 1)
            assign weight_ff2[i*32 +: 32] = parameter_memory[WFF2_OFFSET+i];
        for (i = 0; i < WLM_WORDS; i = i + 1)
            assign weight_lm[i*32 +: 32] = parameter_memory[WLM_OFFSET+i];
        for (i = 0; i < VOCAB_SIZE; i = i + 1)
            assign bias_lm[i*32 +: 32] = parameter_memory[BLM_OFFSET+i];
    endgenerate

    tiny_transformer #(
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
        .cycle_count(cycle_count), .parameter_count(parameter_count)
    );
endmodule

`default_nettype wire
