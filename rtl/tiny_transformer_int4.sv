`timescale 1ns/1ps
`default_nettype none

// Quantized inference-only tiny transformer.
// Model weights and embeddings are signed INT4 (W4), activations are signed
// INT8 (A8), and dot products/biases use signed INT32 accumulation.
// The graph matches tiny_transformer: causal hardmax attention, residuals,
// ReLU feed-forward network, and an argmax language-model head.
module tiny_transformer_int4 #(
    parameter integer VOCAB_SIZE = 256,
    parameter integer SEQ_LEN    = 4,
    parameter integer D_MODEL    = 16,
    parameter integer D_FF       = 32
) (
    input  logic                              clk,
    input  logic                              rst,
    input  logic                              start,
    input  logic [SEQ_LEN*8-1:0]              token_ids,
    input  logic [VOCAB_SIZE*D_MODEL*4-1:0]   token_embedding,
    input  logic [SEQ_LEN*D_MODEL*4-1:0]      position_embedding,
    input  logic [D_MODEL*D_MODEL*4-1:0]      weight_q,
    input  logic [D_MODEL*32-1:0]             bias_q,
    input  logic [D_MODEL*D_MODEL*4-1:0]      weight_k,
    input  logic [D_MODEL*32-1:0]             bias_k,
    input  logic [D_MODEL*D_MODEL*4-1:0]      weight_v,
    input  logic [D_MODEL*32-1:0]             bias_v,
    input  logic [D_MODEL*D_MODEL*4-1:0]      weight_o,
    input  logic [D_MODEL*32-1:0]             bias_o,
    input  logic [D_FF*D_MODEL*4-1:0]         weight_ff1,
    input  logic [D_FF*32-1:0]                bias_ff1,
    input  logic [D_MODEL*D_FF*4-1:0]         weight_ff2,
    input  logic [D_MODEL*32-1:0]             bias_ff2,
    input  logic [VOCAB_SIZE*D_MODEL*4-1:0]   weight_lm,
    input  logic [VOCAB_SIZE*32-1:0]          bias_lm,
    output logic [VOCAB_SIZE*32-1:0]          logits,
    output logic [7:0]                        predicted_token,
    output logic [D_MODEL*8-1:0]              last_hidden,
    output logic                              busy,
    output logic                              done,
    output logic [31:0]                       cycle_count,
    output logic [31:0]                       parameter_count,
    output logic [31:0]                       parameter_bits
);
    localparam integer TOKEN_INDEX_WIDTH =
        (SEQ_LEN <= 1) ? 1 : $clog2(SEQ_LEN);
    localparam logic [TOKEN_INDEX_WIDTH-1:0] LAST_TOKEN =
        TOKEN_INDEX_WIDTH'(SEQ_LEN - 1);

    localparam integer INT4_PARAMETER_COUNT =
        (VOCAB_SIZE*D_MODEL) + (SEQ_LEN*D_MODEL) +
        (4*D_MODEL*D_MODEL) + (D_FF*D_MODEL) +
        (D_MODEL*D_FF) + (VOCAB_SIZE*D_MODEL);
    localparam integer BIAS_PARAMETER_COUNT =
        (5*D_MODEL) + D_FF + VOCAB_SIZE;

    localparam logic [4:0] IDLE       = 5'd0;
    localparam logic [4:0] LOAD_TOKEN = 5'd1;
    localparam logic [4:0] START_QKV  = 5'd2;
    localparam logic [4:0] WAIT_QKV   = 5'd3;
    localparam logic [4:0] START_ATTN = 5'd4;
    localparam logic [4:0] WAIT_ATTN  = 5'd5;
    localparam logic [4:0] STORE_ATTN = 5'd6;
    localparam logic [4:0] START_PROJ = 5'd7;
    localparam logic [4:0] WAIT_PROJ  = 5'd8;
    localparam logic [4:0] START_FF1  = 5'd9;
    localparam logic [4:0] WAIT_FF1   = 5'd10;
    localparam logic [4:0] START_FF2  = 5'd11;
    localparam logic [4:0] WAIT_FF2   = 5'd12;
    localparam logic [4:0] START_LM   = 5'd13;
    localparam logic [4:0] WAIT_LM    = 5'd14;
    localparam logic [4:0] DONE_STATE = 5'd15;

    logic [4:0] state;
    logic [TOKEN_INDEX_WIDTH-1:0] token_index;
    logic [SEQ_LEN*D_MODEL*8-1:0] token_state;
    logic [SEQ_LEN*D_MODEL*8-1:0] query_memory;
    logic [SEQ_LEN*D_MODEL*8-1:0] key_memory;
    logic [SEQ_LEN*D_MODEL*8-1:0] value_memory;
    logic [D_MODEL*8-1:0] context_vector;
    logic [D_MODEL*8-1:0] residual_vector;

    logic [7:0] current_token_id;
    logic [D_MODEL*4-1:0] selected_embedding;
    logic [D_MODEL*8-1:0] embedded_token;
    logic [D_MODEL*8-1:0] current_token_state;
    logic [D_MODEL*8-1:0] current_query;

    logic [D_MODEL*32-1:0] q_raw, k_raw, v_raw;
    logic [D_MODEL*8-1:0] q_result, k_result, v_result;
    logic q_done, k_done, v_done;
    logic q_busy, k_busy, v_busy;
    logic [SEQ_LEN*32-1:0] attention_scores;
    logic attention_done, attention_busy;
    logic [TOKEN_INDEX_WIDTH-1:0] selected_attention_token;
    logic [D_MODEL*32-1:0] projection_raw;
    logic [D_MODEL*8-1:0] projection_result;
    logic projection_done, projection_busy;
    logic [D_MODEL*8-1:0] projected_residual;
    logic [D_FF*32-1:0] ff1_raw;
    logic [D_FF*8-1:0] ff1_relu;
    logic ff1_done, ff1_busy;
    logic [D_MODEL*32-1:0] ff2_raw;
    logic [D_MODEL*8-1:0] ff2_result;
    logic ff2_done, ff2_busy;
    logic [D_MODEL*8-1:0] final_residual;
    logic lm_done, lm_busy;
    logic signed [31:0] largest_attention_score;
    logic signed [31:0] largest_logit;
    integer attention_compare_index;
    integer logit_compare_index;

    function automatic logic [7:0] saturate_int8(
        input logic signed [31:0] value
    );
        begin
            if (value > 32'sd127)
                saturate_int8 = 8'h7f;
            else if (value < -32'sd128)
                saturate_int8 = 8'h80;
            else
                saturate_int8 = value[7:0];
        end
    endfunction

    function automatic logic [7:0] saturating_add_int8(
        input logic [7:0] a,
        input logic [7:0] b
    );
        logic signed [8:0] sum;
        begin
            sum = $signed(a) + $signed(b);
            if (sum > 9'sd127)
                saturating_add_int8 = 8'h7f;
            else if (sum < -9'sd128)
                saturating_add_int8 = 8'h80;
            else
                saturating_add_int8 = sum[7:0];
        end
    endfunction

    assign parameter_count = INT4_PARAMETER_COUNT + BIAS_PARAMETER_COUNT;
    assign parameter_bits = (INT4_PARAMETER_COUNT*4) +
                            (BIAS_PARAMETER_COUNT*32);
    assign current_token_id = token_ids[token_index*8 +: 8];
    assign selected_embedding = token_embedding[
        (32'(current_token_id)*D_MODEL)*4 +: D_MODEL*4];
    assign current_token_state = token_state[
        (32'(token_index)*D_MODEL)*8 +: D_MODEL*8];
    assign current_query = query_memory[
        (32'(token_index)*D_MODEL)*8 +: D_MODEL*8];
    assign last_hidden = token_state[
        ((SEQ_LEN-1)*D_MODEL)*8 +: D_MODEL*8];

    genvar dimension;
    generate
        for (dimension = 0; dimension < D_MODEL;
             dimension = dimension + 1) begin : model_lane
            logic signed [4:0] embedding_sum;
            assign embedding_sum =
                $signed(selected_embedding[dimension*4 +: 4]) +
                $signed(position_embedding[
                    (32'(token_index)*D_MODEL + dimension)*4 +: 4]);
            assign embedded_token[dimension*8 +: 8] =
                {{3{embedding_sum[4]}}, embedding_sum};
            assign q_result[dimension*8 +: 8] =
                saturate_int8($signed(q_raw[dimension*32 +: 32]));
            assign k_result[dimension*8 +: 8] =
                saturate_int8($signed(k_raw[dimension*32 +: 32]));
            assign v_result[dimension*8 +: 8] =
                saturate_int8($signed(v_raw[dimension*32 +: 32]));
            assign projection_result[dimension*8 +: 8] =
                saturate_int8($signed(projection_raw[dimension*32 +: 32]));
            assign projected_residual[dimension*8 +: 8] =
                saturating_add_int8(
                    current_token_state[dimension*8 +: 8],
                    projection_result[dimension*8 +: 8]);
            assign ff2_result[dimension*8 +: 8] =
                saturate_int8($signed(ff2_raw[dimension*32 +: 32]));
            assign final_residual[dimension*8 +: 8] =
                saturating_add_int8(
                    residual_vector[dimension*8 +: 8],
                    ff2_result[dimension*8 +: 8]);
        end
    endgenerate

    genvar ff_index;
    generate
        for (ff_index = 0; ff_index < D_FF;
             ff_index = ff_index + 1) begin : ff_relu_lane
            assign ff1_relu[ff_index*8 +: 8] =
                $signed(ff1_raw[ff_index*32 +: 32]) <= 0 ? 8'd0 :
                saturate_int8($signed(ff1_raw[ff_index*32 +: 32]));
        end
    endgenerate

    int4_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) q_projection (
        .clk(clk), .rst(rst), .start(state == START_QKV),
        .input_vector(current_token_state), .weights(weight_q), .bias(bias_q),
        .output_vector(q_raw), .busy(q_busy), .done(q_done)
    );
    int4_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) k_projection (
        .clk(clk), .rst(rst), .start(state == START_QKV),
        .input_vector(current_token_state), .weights(weight_k), .bias(bias_k),
        .output_vector(k_raw), .busy(k_busy), .done(k_done)
    );
    int4_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) v_projection (
        .clk(clk), .rst(rst), .start(state == START_QKV),
        .input_vector(current_token_state), .weights(weight_v), .bias(bias_v),
        .output_vector(v_raw), .busy(v_busy), .done(v_done)
    );

    int8_dot_bank #(.IN_SIZE(D_MODEL), .OUT_SIZE(SEQ_LEN)) attention_dots (
        .clk(clk), .rst(rst), .start(state == START_ATTN),
        .input_vector(current_query), .vectors(key_memory),
        .scores(attention_scores), .busy(attention_busy),
        .done(attention_done)
    );

    int4_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_MODEL)) output_projection (
        .clk(clk), .rst(rst), .start(state == START_PROJ),
        .input_vector(context_vector), .weights(weight_o), .bias(bias_o),
        .output_vector(projection_raw), .busy(projection_busy),
        .done(projection_done)
    );
    int4_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(D_FF)) feed_forward_1 (
        .clk(clk), .rst(rst), .start(state == START_FF1),
        .input_vector(residual_vector), .weights(weight_ff1), .bias(bias_ff1),
        .output_vector(ff1_raw), .busy(ff1_busy), .done(ff1_done)
    );
    int4_linear #(.IN_SIZE(D_FF), .OUT_SIZE(D_MODEL)) feed_forward_2 (
        .clk(clk), .rst(rst), .start(state == START_FF2),
        .input_vector(ff1_relu), .weights(weight_ff2), .bias(bias_ff2),
        .output_vector(ff2_raw), .busy(ff2_busy), .done(ff2_done)
    );
    int4_linear #(.IN_SIZE(D_MODEL), .OUT_SIZE(VOCAB_SIZE)) lm_head (
        .clk(clk), .rst(rst), .start(state == START_LM),
        .input_vector(last_hidden), .weights(weight_lm), .bias(bias_lm),
        .output_vector(logits), .busy(lm_busy), .done(lm_done)
    );

    always @* begin
        selected_attention_token = 0;
        largest_attention_score = $signed(attention_scores[0 +: 32]);
        for (attention_compare_index = 1;
             attention_compare_index < SEQ_LEN;
             attention_compare_index = attention_compare_index + 1) begin
            if ((attention_compare_index <= 32'(token_index)) &&
                ($signed(attention_scores[
                    attention_compare_index*32 +: 32]) >
                 largest_attention_score)) begin
                largest_attention_score = $signed(attention_scores[
                    attention_compare_index*32 +: 32]);
                selected_attention_token =
                    TOKEN_INDEX_WIDTH'(attention_compare_index);
            end
        end
    end

    always @* begin
        predicted_token = 0;
        largest_logit = $signed(logits[0 +: 32]);
        for (logit_compare_index = 1; logit_compare_index < VOCAB_SIZE;
             logit_compare_index = logit_compare_index + 1) begin
            if ($signed(logits[logit_compare_index*32 +: 32]) >
                largest_logit) begin
                largest_logit =
                    $signed(logits[logit_compare_index*32 +: 32]);
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
                    token_state[(32'(token_index)*D_MODEL)*8 +: D_MODEL*8]
                        <= embedded_token;
                    if (token_index == LAST_TOKEN) begin
                        token_index <= 0;
                        state <= START_QKV;
                    end else
                        token_index <= token_index + 1'b1;
                end
                START_QKV: state <= WAIT_QKV;
                WAIT_QKV: if (q_done && k_done && v_done) begin
                    query_memory[(32'(token_index)*D_MODEL)*8 +: D_MODEL*8]
                        <= q_result;
                    key_memory[(32'(token_index)*D_MODEL)*8 +: D_MODEL*8]
                        <= k_result;
                    value_memory[(32'(token_index)*D_MODEL)*8 +: D_MODEL*8]
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
                        (32'(selected_attention_token)*D_MODEL)*8 +:
                        D_MODEL*8];
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
                    token_state[(32'(token_index)*D_MODEL)*8 +: D_MODEL*8]
                        <= final_residual;
                    if (token_index == LAST_TOKEN)
                        state <= START_LM;
                    else begin
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
