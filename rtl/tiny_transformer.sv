`timescale 1ns/1ps
`default_nettype none

// Tiny inference-only FP32 transformer-like language model.
// Default parameter count: 10,672 trainable FP32 values.
//
// Architecture:
//   token + position embedding
//   -> Q/K/V projections
//   -> single-head causal hardmax self-attention
//   -> output projection + residual
//   -> 16-to-32-to-16 ReLU feed-forward + residual
//   -> 256-token language-model head + argmax
//
// Hardmax attention selects the allowed V vector with the largest Q dot K
// score. It replaces exponential softmax, and layer normalization is omitted,
// so this is a compact transformer variant rather than a standard GPT block.
module tiny_transformer #(
    parameter integer VOCAB_SIZE = 256,
    parameter integer SEQ_LEN    = 4,
    parameter integer D_MODEL    = 16,
    parameter integer D_FF       = 32
) (
    input  logic                                  clk,
    input  logic                                  rst,
    input  logic                                  start,
    input  logic [SEQ_LEN*8-1:0]                  token_ids,
    input  logic [VOCAB_SIZE*D_MODEL*32-1:0]      token_embedding,
    input  logic [SEQ_LEN*D_MODEL*32-1:0]         position_embedding,
    input  logic [D_MODEL*D_MODEL*32-1:0]         weight_q,
    input  logic [D_MODEL*32-1:0]                 bias_q,
    input  logic [D_MODEL*D_MODEL*32-1:0]         weight_k,
    input  logic [D_MODEL*32-1:0]                 bias_k,
    input  logic [D_MODEL*D_MODEL*32-1:0]         weight_v,
    input  logic [D_MODEL*32-1:0]                 bias_v,
    input  logic [D_MODEL*D_MODEL*32-1:0]         weight_o,
    input  logic [D_MODEL*32-1:0]                 bias_o,
    input  logic [D_FF*D_MODEL*32-1:0]            weight_ff1,
    input  logic [D_FF*32-1:0]                    bias_ff1,
    input  logic [D_MODEL*D_FF*32-1:0]            weight_ff2,
    input  logic [D_MODEL*32-1:0]                 bias_ff2,
    input  logic [VOCAB_SIZE*D_MODEL*32-1:0]      weight_lm,
    input  logic [VOCAB_SIZE*32-1:0]              bias_lm,
    output logic [VOCAB_SIZE*32-1:0]              logits,
    output logic [7:0]                            predicted_token,
    output logic [D_MODEL*32-1:0]                 last_hidden,
    output logic                                  busy,
    output logic                                  done,
    output logic [31:0]                           cycle_count,
    output logic [31:0]                           parameter_count
);
    localparam integer TOKEN_INDEX_WIDTH =
        (SEQ_LEN <= 1) ? 1 : $clog2(SEQ_LEN);
    localparam logic [TOKEN_INDEX_WIDTH-1:0] LAST_TOKEN =
        TOKEN_INDEX_WIDTH'(SEQ_LEN - 1);

    localparam logic [4:0] IDLE          = 5'd0;
    localparam logic [4:0] LOAD_TOKEN    = 5'd1;
    localparam logic [4:0] START_QKV     = 5'd2;
    localparam logic [4:0] WAIT_QKV      = 5'd3;
    localparam logic [4:0] START_ATTN    = 5'd4;
    localparam logic [4:0] WAIT_ATTN     = 5'd5;
    localparam logic [4:0] STORE_ATTN    = 5'd6;
    localparam logic [4:0] START_PROJ    = 5'd7;
    localparam logic [4:0] WAIT_PROJ     = 5'd8;
    localparam logic [4:0] START_FF1     = 5'd9;
    localparam logic [4:0] WAIT_FF1      = 5'd10;
    localparam logic [4:0] START_FF2     = 5'd11;
    localparam logic [4:0] WAIT_FF2      = 5'd12;
    localparam logic [4:0] START_LM      = 5'd13;
    localparam logic [4:0] WAIT_LM       = 5'd14;
    localparam logic [4:0] DONE_STATE    = 5'd15;

    logic [4:0] state;
    logic [TOKEN_INDEX_WIDTH-1:0] token_index;

    logic [SEQ_LEN*D_MODEL*32-1:0] token_state;
    logic [SEQ_LEN*D_MODEL*32-1:0] query_memory;
    logic [SEQ_LEN*D_MODEL*32-1:0] key_memory;
    logic [SEQ_LEN*D_MODEL*32-1:0] value_memory;
    logic [D_MODEL*32-1:0] context_vector;
    logic [D_MODEL*32-1:0] residual_vector;

    logic [7:0] current_token_id;
    logic [D_MODEL*32-1:0] selected_embedding;
    logic [D_MODEL*32-1:0] embedded_token;
    logic [D_MODEL*32-1:0] current_token_state;
    logic [D_MODEL*32-1:0] current_query;

    logic [D_MODEL*32-1:0] q_result, k_result, v_result;
    logic q_done, k_done, v_done;
    logic [SEQ_LEN*32-1:0] attention_scores;
    logic attention_done;
    logic [TOKEN_INDEX_WIDTH-1:0] selected_attention_token;
    logic [D_MODEL*32-1:0] projection_result;
    logic projection_done;
    logic [D_MODEL*32-1:0] projected_residual;
    logic [D_FF*32-1:0] ff1_result;
    logic [D_FF*32-1:0] ff1_relu;
    logic ff1_done;
    logic [D_MODEL*32-1:0] ff2_result;
    logic ff2_done;
    logic [D_MODEL*32-1:0] final_residual;
    logic lm_done;

    logic q_busy, k_busy, v_busy, attention_busy;
    logic projection_busy, ff1_busy, ff2_busy, lm_busy;
    logic [31:0] largest_attention_score;
    logic [31:0] largest_logit;
    integer attention_compare_index;
    integer logit_compare_index;

    assign parameter_count =
        (VOCAB_SIZE*D_MODEL) + (SEQ_LEN*D_MODEL) +
        (3*(D_MODEL*D_MODEL + D_MODEL)) +
        (D_MODEL*D_MODEL + D_MODEL) +
        (D_FF*D_MODEL + D_FF) + (D_MODEL*D_FF + D_MODEL) +
        (VOCAB_SIZE*D_MODEL + VOCAB_SIZE);

    assign current_token_id = token_ids[token_index*8 +: 8];
    assign selected_embedding =
        token_embedding[(32'(current_token_id)*D_MODEL)*32 +: D_MODEL*32];
    assign current_token_state =
        token_state[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32];
    assign current_query =
        query_memory[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32];
    assign last_hidden =
        token_state[((SEQ_LEN-1)*D_MODEL)*32 +: D_MODEL*32];

    function automatic fp32_greater_than;
        input logic [31:0] a;
        input logic [31:0] b;
        logic a_nan, b_nan;
        begin
            a_nan = (a[30:23] == 8'hff) && (a[22:0] != 0);
            b_nan = (b[30:23] == 8'hff) && (b[22:0] != 0);
            if (a_nan)
                fp32_greater_than = 1'b0;
            else if (b_nan)
                fp32_greater_than = 1'b1;
            else if ((a[30:0] == 0) && (b[30:0] == 0))
                fp32_greater_than = 1'b0;
            else if (a[31] != b[31])
                fp32_greater_than = !a[31];
            else if (!a[31])
                fp32_greater_than = a[30:0] > b[30:0];
            else
                fp32_greater_than = a[30:0] < b[30:0];
        end
    endfunction

    genvar dimension;
    generate
        for (dimension = 0; dimension < D_MODEL;
             dimension = dimension + 1) begin : vector_arithmetic
            fp32_add embed_add (
                .a(selected_embedding[dimension*32 +: 32]),
                .b(position_embedding[(32'(token_index)*D_MODEL + dimension)*32 +: 32]),
                .result(embedded_token[dimension*32 +: 32])
            );
            fp32_add projection_residual_add (
                .a(current_token_state[dimension*32 +: 32]),
                .b(projection_result[dimension*32 +: 32]),
                .result(projected_residual[dimension*32 +: 32])
            );
            fp32_add final_residual_add (
                .a(residual_vector[dimension*32 +: 32]),
                .b(ff2_result[dimension*32 +: 32]),
                .result(final_residual[dimension*32 +: 32])
            );
        end
    endgenerate

    genvar ff_index;
    generate
        for (ff_index = 0; ff_index < D_FF; ff_index = ff_index + 1) begin : ff_relu_lane
            fp32_relu relu (
                .value(ff1_result[ff_index*32 +: 32]),
                .result(ff1_relu[ff_index*32 +: 32])
            );
        end
    endgenerate

    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) q_projection (
        .clk(clk), .rst(rst), .start(state == START_QKV),
        .input_vector(current_token_state), .weights(weight_q), .bias(bias_q),
        .output_vector(q_result), .busy(q_busy), .done(q_done)
    );
    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) k_projection (
        .clk(clk), .rst(rst), .start(state == START_QKV),
        .input_vector(current_token_state), .weights(weight_k), .bias(bias_k),
        .output_vector(k_result), .busy(k_busy), .done(k_done)
    );
    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) v_projection (
        .clk(clk), .rst(rst), .start(state == START_QKV),
        .input_vector(current_token_state), .weights(weight_v), .bias(bias_v),
        .output_vector(v_result), .busy(v_busy), .done(v_done)
    );

    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(SEQ_LEN)) attention_dot_products (
        .clk(clk), .rst(rst), .start(state == START_ATTN),
        .input_vector(current_query), .weights(key_memory),
        .bias({SEQ_LEN*32{1'b0}}), .output_vector(attention_scores),
        .busy(attention_busy), .done(attention_done)
    );

    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) output_projection (
        .clk(clk), .rst(rst), .start(state == START_PROJ),
        .input_vector(context_vector), .weights(weight_o), .bias(bias_o),
        .output_vector(projection_result),
        .busy(projection_busy), .done(projection_done)
    );

    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_FF)) feed_forward_1 (
        .clk(clk), .rst(rst), .start(state == START_FF1),
        .input_vector(residual_vector), .weights(weight_ff1), .bias(bias_ff1),
        .output_vector(ff1_result), .busy(ff1_busy), .done(ff1_done)
    );

    fp32_linear #(.IN_SIZE(D_FF), .OUT_SIZE(D_MODEL)) feed_forward_2 (
        .clk(clk), .rst(rst), .start(state == START_FF2),
        .input_vector(ff1_relu), .weights(weight_ff2), .bias(bias_ff2),
        .output_vector(ff2_result), .busy(ff2_busy), .done(ff2_done)
    );

    fp32_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(VOCAB_SIZE)) language_model_head (
        .clk(clk), .rst(rst), .start(state == START_LM),
        .input_vector(last_hidden), .weights(weight_lm), .bias(bias_lm),
        .output_vector(logits), .busy(lm_busy), .done(lm_done)
    );

    always @* begin
        selected_attention_token = 0;
        largest_attention_score = attention_scores[0 +: 32];
        for (attention_compare_index = 1;
             attention_compare_index < SEQ_LEN;
             attention_compare_index = attention_compare_index + 1) begin
            if ((attention_compare_index <= 32'(token_index)) &&
                fp32_greater_than(
                    attention_scores[attention_compare_index*32 +: 32],
                                  largest_attention_score)) begin
                largest_attention_score =
                    attention_scores[attention_compare_index*32 +: 32];
                selected_attention_token =
                    TOKEN_INDEX_WIDTH'(attention_compare_index);
            end
        end
    end

    always @* begin
        predicted_token = 0;
        largest_logit = logits[0 +: 32];
        for (logit_compare_index = 1; logit_compare_index < VOCAB_SIZE;
             logit_compare_index = logit_compare_index + 1) begin
            if (fp32_greater_than(logits[logit_compare_index*32 +: 32],
                                  largest_logit)) begin
                largest_logit = logits[logit_compare_index*32 +: 32];
                predicted_token = 8'(logit_compare_index);
            end
        end
    end

    assign busy = ((state != IDLE) && (state != DONE_STATE)) |
                  q_busy | k_busy | v_busy | attention_busy |
                  projection_busy | ff1_busy | ff2_busy | lm_busy;
    assign done = (state == DONE_STATE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            token_index <= 0;
            token_state <= 0;
            query_memory <= 0;
            key_memory <= 0;
            value_memory <= 0;
            context_vector <= 0;
            residual_vector <= 0;
            cycle_count <= 0;
        end else begin
            if (state != IDLE && state != DONE_STATE)
                cycle_count <= cycle_count + 1;

            case (state)
                IDLE: begin
                    token_index <= 0;
                    if (start) begin
                        cycle_count <= 0;
                        state <= LOAD_TOKEN;
                    end
                end

                LOAD_TOKEN: begin
                    token_state[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32]
                        <= embedded_token;
                    if (token_index == LAST_TOKEN) begin
                        token_index <= 0;
                        state <= START_QKV;
                    end else begin
                        token_index <= token_index + 1'b1;
                    end
                end

                START_QKV: state <= WAIT_QKV;
                WAIT_QKV: if (q_done && k_done && v_done) begin
                    query_memory[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32]
                        <= q_result;
                    key_memory[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32]
                        <= k_result;
                    value_memory[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32]
                        <= v_result;
                    if (token_index == LAST_TOKEN) begin
                        token_index <= 0;
                        state <= START_ATTN;
                    end else begin
                        token_index <= token_index + 1'b1;
                        state <= START_QKV;
                    end
                end

                START_ATTN: state <= WAIT_ATTN;
                WAIT_ATTN: if (attention_done)
                    state <= STORE_ATTN;
                STORE_ATTN: begin
                    context_vector <= value_memory[
                        (32'(selected_attention_token)*D_MODEL)*32 +: D_MODEL*32];
                    state <= START_PROJ;
                end

                START_PROJ: state <= WAIT_PROJ;
                WAIT_PROJ: if (projection_done) begin
                    residual_vector <= projected_residual;
                    state <= START_FF1;
                end

                START_FF1: state <= WAIT_FF1;
                WAIT_FF1: if (ff1_done)
                    state <= START_FF2;

                START_FF2: state <= WAIT_FF2;
                WAIT_FF2: if (ff2_done) begin
                    token_state[(32'(token_index)*D_MODEL)*32 +: D_MODEL*32]
                        <= final_residual;
                    if (token_index == LAST_TOKEN) begin
                        state <= START_LM;
                    end else begin
                        token_index <= token_index + 1'b1;
                        state <= START_ATTN;
                    end
                end

                START_LM: state <= WAIT_LM;
                WAIT_LM: if (lm_done)
                    state <= DONE_STATE;
                DONE_STATE: state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
