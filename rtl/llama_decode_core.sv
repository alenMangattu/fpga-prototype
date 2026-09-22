`timescale 1ns/1ps
`default_nettype none

// Standalone, synthesizable, decoder-only Llama execution core.
//
// All inference arithmetic and scheduling are RTL.  The configuration port is
// used only to load fixed-point model constants before inference.  Values use
// signed Q(FRAC) fixed point.  The deliberately serial datapath makes the RTL
// practical to verify and synthesize at small sizes; a production part should
// replicate the MAC and replace parameter_memory/KV arrays with memory macros.
//
// Implemented graph per layer:
//   RMSNorm -> Q/K/V -> RoPE -> GQA softmax attention -> output projection
//   -> residual -> RMSNorm -> SwiGLU -> down projection -> residual.
// The core also performs embedding lookup, final RMSNorm, tied output
// projection, argmax token selection, layer scheduling, and persistent KV.
module llama_decode_core #(
    parameter integer D_MODEL = 8,
    parameter integer D_FF = 16,
    parameter integer HEADS = 2,
    parameter integer KV_HEADS = 1,
    parameter integer LAYERS = 2,
    parameter integer VOCAB_SIZE = 16,
    parameter integer MAX_CONTEXT = 8,
    parameter integer FRAC = 16,
    parameter integer CFG_ADDR_BITS = 31
) (
    input  wire                         clk,
    input  wire                         reset_n,
    input  wire                         clear_cache,
    input  wire                         start,
    input  wire [$clog2(VOCAB_SIZE)-1:0] input_token,
    input  wire                         cfg_we,
    input  wire [CFG_ADDR_BITS-1:0]     cfg_addr,
    input  wire [31:0]                  cfg_wdata,
    output reg  [31:0]                  cfg_rdata,
    output wire                         cfg_ready,
    output reg                          busy,
    output reg                          done,
    output reg  [$clog2(VOCAB_SIZE)-1:0] output_token,
    output reg  [$clog2(MAX_CONTEXT+1)-1:0] position,
    output reg  [31:0]                  cycle_count,
    output wire [31:0]                  parameter_words
);
    localparam integer HEAD_DIM = D_MODEL / HEADS;
    localparam integer KV_DIM = KV_HEADS * HEAD_DIM;
    localparam integer GROUPS = HEADS / KV_HEADS;
    localparam integer PAIRS_PER_POS = HEAD_DIM / 2;

    // Global constants and flattened row-major tensor map.
    localparam integer RMS_EPS_ADDR = 0;
    localparam integer ATTN_SCALE_ADDR = 1;
    localparam integer EMBED_BASE = 2;
    localparam integer EMBED_WORDS = VOCAB_SIZE * D_MODEL;
    localparam integer ROPE_COS_BASE = EMBED_BASE + EMBED_WORDS;
    localparam integer ROPE_WORDS = MAX_CONTEXT * PAIRS_PER_POS;
    localparam integer ROPE_SIN_BASE = ROPE_COS_BASE + ROPE_WORDS;
    localparam integer LAYER_BASE = ROPE_SIN_BASE + ROPE_WORDS;
    localparam integer LAYER_WORDS =
        D_MODEL + D_MODEL*D_MODEL + KV_DIM*D_MODEL + KV_DIM*D_MODEL +
        D_MODEL*D_MODEL + D_MODEL + D_FF*D_MODEL + D_FF*D_MODEL +
        D_MODEL*D_FF;
    localparam integer FINAL_NORM_BASE = LAYER_BASE + LAYERS*LAYER_WORDS;
    localparam integer PARAM_WORDS = FINAL_NORM_BASE + D_MODEL;

    localparam integer S_IDLE=0, S_EMBED=1, S_NORM_SQ=2, S_NORM_OUT=3,
        S_GEMV=4, S_ROPE_Q=5, S_ROPE_K=6, S_KV_STORE=7,
        S_SCORE_INIT=8, S_SCORE_DOT=9, S_SOFTMAX_MAX=10,
        S_SOFTMAX_EXP=11, S_SOFTMAX_DIV=12, S_ATTN_INIT=13,
        S_ATTN_SUM=14, S_RESID_ATTN=15, S_SWIGLU=16,
        S_RESID_FF=17, S_ARGMAX=18, S_FINISH=19;
    localparam integer NORM_ATTN=0, NORM_FFN=1, NORM_FINAL=2;
    localparam integer OP_Q=0, OP_K=1, OP_V=2, OP_O=3,
        OP_GATE=4, OP_UP=5, OP_DOWN=6, OP_LOGITS=7;

    reg signed [31:0] parameter_memory [0:PARAM_WORDS-1];
    reg signed [31:0] x [0:D_MODEL-1];
    reg signed [31:0] normalized [0:D_MODEL-1];
    reg signed [31:0] q [0:D_MODEL-1];
    reg signed [31:0] k [0:KV_DIM-1];
    reg signed [31:0] v [0:KV_DIM-1];
    reg signed [31:0] attention [0:D_MODEL-1];
    reg signed [31:0] projected [0:D_MODEL-1];
    reg signed [31:0] gate [0:D_FF-1];
    reg signed [31:0] up [0:D_FF-1];
    reg signed [31:0] ff [0:D_FF-1];
    reg signed [31:0] kv_key [0:LAYERS*MAX_CONTEXT*KV_DIM-1];
    reg signed [31:0] kv_value [0:LAYERS*MAX_CONTEXT*KV_DIM-1];
    reg signed [31:0] scores [0:HEADS*MAX_CONTEXT-1];
    reg signed [31:0] probabilities [0:HEADS*MAX_CONTEXT-1];

    integer state, norm_kind, gemv_op;
    integer layer_index, index, row, column, head, time_index;
    integer pair_index;
    integer matrix_base, matrix_rows, matrix_cols;
    reg signed [63:0] norm_sum;
    reg signed [31:0] norm_factor, mac_acc, softmax_max, softmax_sum;
    reg signed [31:0] best_logit;
    reg [$clog2(VOCAB_SIZE)-1:0] best_token;

    assign cfg_ready = !busy;
    assign parameter_words = PARAM_WORDS;

    function automatic integer layer_offset(input integer layer_no);
        layer_offset = LAYER_BASE + layer_no*LAYER_WORDS;
    endfunction
    function automatic integer attn_norm_base(input integer layer_no);
        attn_norm_base = layer_offset(layer_no);
    endfunction
    function automatic integer q_base(input integer layer_no);
        q_base = attn_norm_base(layer_no) + D_MODEL;
    endfunction
    function automatic integer k_base(input integer layer_no);
        k_base = q_base(layer_no) + D_MODEL*D_MODEL;
    endfunction
    function automatic integer v_base(input integer layer_no);
        v_base = k_base(layer_no) + KV_DIM*D_MODEL;
    endfunction
    function automatic integer o_base(input integer layer_no);
        o_base = v_base(layer_no) + KV_DIM*D_MODEL;
    endfunction
    function automatic integer ffn_norm_base(input integer layer_no);
        ffn_norm_base = o_base(layer_no) + D_MODEL*D_MODEL;
    endfunction
    function automatic integer gate_base(input integer layer_no);
        gate_base = ffn_norm_base(layer_no) + D_MODEL;
    endfunction
    function automatic integer up_base(input integer layer_no);
        up_base = gate_base(layer_no) + D_FF*D_MODEL;
    endfunction
    function automatic integer down_base(input integer layer_no);
        down_base = up_base(layer_no) + D_FF*D_MODEL;
    endfunction

    function automatic signed [31:0] sat32(input signed [63:0] value);
        begin
            if (value > 64'sh0000_0000_7fff_ffff)
                sat32 = 32'sh7fff_ffff;
            else if (value < -64'sh0000_0000_8000_0000)
                sat32 = -32'sh8000_0000;
            else
                sat32 = value[31:0];
        end
    endfunction

    function automatic signed [31:0] qadd(
        input signed [31:0] a, input signed [31:0] b);
        reg signed [32:0] sum;
        begin
            sum = {a[31],a} + {b[31],b};
            qadd = sat32({{31{sum[32]}},sum});
        end
    endfunction

    function automatic signed [31:0] qmul(
        input signed [31:0] a, input signed [31:0] b);
        reg signed [63:0] product;
        reg signed [63:0] rounded;
        begin
            product = a*b;
            if (FRAC > 0) begin
                if (product >= 0) begin
                    rounded = (product + (64'sd1 << (FRAC-1))) >>> FRAC;
                end else begin
                    rounded = -(((-product) + (64'sd1 << (FRAC-1))) >>> FRAC);
                end
                qmul = sat32(rounded);
            end else begin
                qmul = sat32(product);
            end
        end
    endfunction

    function automatic signed [31:0] qdiv(
        input signed [31:0] numerator, input signed [31:0] denominator);
        reg signed [63:0] wide;
        begin
            if (denominator == 0)
                qdiv = numerator[31] ? -32'sh7fff_ffff : 32'sh7fff_ffff;
            else begin
                wide = {{32{numerator[31]}},numerator};
                wide = wide <<< FRAC;
                qdiv = sat32(wide / {{32{denominator[31]}},denominator});
            end
        end
    endfunction

    function automatic [63:0] isqrt64(input [63:0] value);
        reg [63:0] result_value, bit_value, trial;
        integer n;
        begin
            result_value = 0;
            bit_value = 64'h4000_0000_0000_0000;
            for (n=0; n<32; n=n+1) begin
                trial = result_value + bit_value;
                if (value >= trial) begin
                    value = value - trial;
                    result_value = (result_value >> 1) + bit_value;
                end else begin
                    result_value = result_value >> 1;
                end
                bit_value = bit_value >> 2;
            end
            isqrt64 = result_value;
        end
    endfunction

    function automatic signed [31:0] qsqrt(input signed [31:0] value);
        reg [63:0] shifted;
        reg [63:0] root;
        begin
            if (value <= 0)
                qsqrt = 0;
            else begin
                shifted = {32'd0,$unsigned(value)};
                shifted = shifted << FRAC;
                root = isqrt64(shifted);
                qsqrt = root[31:0];
            end
        end
    endfunction

    // Range-limited synthesizable exponential.  Softmax supplies x<=0; SiLU
    // clamps -x to [-8,8].  (1+x/256)^256 gives a deterministic approximation.
    function automatic signed [31:0] qexp(input signed [31:0] value);
        reg signed [31:0] clipped, term;
        integer n;
        begin
            if (value > (8 <<< FRAC)) clipped = (8 <<< FRAC);
            else if (value < -(8 <<< FRAC)) clipped = -(8 <<< FRAC);
            else clipped = value;
            term = (1 <<< FRAC) + (clipped >>> 8);
            for (n=0; n<8; n=n+1)
                term = qmul(term, term);
            qexp = term;
        end
    endfunction

    function automatic signed [31:0] qsilu(input signed [31:0] value);
        reg signed [31:0] denom;
        begin
            denom = qadd(1 <<< FRAC, qexp(-value));
            qsilu = qdiv(value, denom);
        end
    endfunction

    function automatic signed [31:0] source_value(input integer col);
        begin
            case (gemv_op)
                OP_Q, OP_K, OP_V, OP_GATE, OP_UP: source_value = normalized[col];
                OP_O: source_value = attention[col];
                OP_DOWN: source_value = ff[col];
                default: source_value = normalized[col];
            endcase
        end
    endfunction

    task automatic begin_gemv(
        input integer operation, input integer base,
        input integer rows_value, input integer cols_value);
        begin
            gemv_op <= operation;
            matrix_base <= base;
            matrix_rows <= rows_value;
            matrix_cols <= cols_value;
            row <= 0;
            column <= 0;
            mac_acc <= 0;
            state <= S_GEMV;
        end
    endtask

    always @* begin
        if (cfg_addr < PARAM_WORDS)
            cfg_rdata = parameter_memory[cfg_addr];
        else
            cfg_rdata = 0;
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE;
            busy <= 0;
            done <= 0;
            output_token <= 0;
            position <= 0;
            cycle_count <= 0;
            layer_index <= 0;
            index <= 0;
        end else begin
            done <= 0;
            if (clear_cache && !busy) begin
                // Position validity, rather than clearing every bit, allows
                // the KV arrays to map to dense SRAM macros.
                position <= 0;
            end
            if (cfg_we && cfg_ready && cfg_addr < PARAM_WORDS)
                parameter_memory[cfg_addr] <= cfg_wdata;
            if (busy)
                cycle_count <= cycle_count + 1;

            case (state)
                S_IDLE: if (start && !cfg_we && position < MAX_CONTEXT) begin
                    busy <= 1;
                    cycle_count <= 0;
                    layer_index <= 0;
                    index <= 0;
                    state <= S_EMBED;
                end

                S_EMBED: begin
                    x[index] <= parameter_memory[EMBED_BASE + input_token*D_MODEL + index];
                    if (index == D_MODEL-1) begin
                        index <= 0;
                        norm_kind <= NORM_ATTN;
                        norm_sum <= 0;
                        state <= S_NORM_SQ;
                    end else index <= index + 1;
                end

                S_NORM_SQ: begin
                    norm_sum <= norm_sum + qmul(x[index], x[index]);
                    if (index == D_MODEL-1) begin
                        // Include the current element because norm_sum updates at edge.
                        norm_factor <= qdiv(1 <<< FRAC, qsqrt(
                            sat32((norm_sum + qmul(x[index],x[index])) / D_MODEL)
                            + parameter_memory[RMS_EPS_ADDR]));
                        index <= 0;
                        state <= S_NORM_OUT;
                    end else index <= index + 1;
                end

                S_NORM_OUT: begin
                    if (norm_kind == NORM_ATTN)
                        normalized[index] <= qmul(qmul(x[index], norm_factor),
                            parameter_memory[attn_norm_base(layer_index)+index]);
                    else if (norm_kind == NORM_FFN)
                        normalized[index] <= qmul(qmul(x[index], norm_factor),
                            parameter_memory[ffn_norm_base(layer_index)+index]);
                    else
                        normalized[index] <= qmul(qmul(x[index], norm_factor),
                            parameter_memory[FINAL_NORM_BASE+index]);
                    if (index == D_MODEL-1) begin
                        index <= 0;
                        if (norm_kind == NORM_ATTN)
                            begin_gemv(OP_Q, q_base(layer_index), D_MODEL, D_MODEL);
                        else if (norm_kind == NORM_FFN)
                            begin_gemv(OP_GATE, gate_base(layer_index), D_FF, D_MODEL);
                        else
                            begin_gemv(OP_LOGITS, EMBED_BASE, VOCAB_SIZE, D_MODEL);
                    end else index <= index + 1;
                end

                S_GEMV: begin : gemv_step
                    reg signed [31:0] next_acc;
                    next_acc = qadd(mac_acc, qmul(source_value(column),
                        parameter_memory[matrix_base + row*matrix_cols + column]));
                    if (column == matrix_cols-1) begin
                        case (gemv_op)
                            OP_Q: q[row] <= next_acc;
                            OP_K: k[row] <= next_acc;
                            OP_V: v[row] <= next_acc;
                            OP_O: projected[row] <= next_acc;
                            OP_GATE: gate[row] <= next_acc;
                            OP_UP: up[row] <= next_acc;
                            OP_DOWN: projected[row] <= next_acc;
                            OP_LOGITS: begin
                                if ((row == 0) || (next_acc > best_logit)) begin
                                    best_logit <= next_acc;
                                    best_token <= row[$clog2(VOCAB_SIZE)-1:0];
                                end
                            end
                        endcase
                        column <= 0;
                        mac_acc <= 0;
                        if (row == matrix_rows-1) begin
                            row <= 0;
                            case (gemv_op)
                                OP_Q: begin_gemv(OP_K, k_base(layer_index), KV_DIM, D_MODEL);
                                OP_K: begin_gemv(OP_V, v_base(layer_index), KV_DIM, D_MODEL);
                                OP_V: begin
                                    head <= 0; pair_index <= 0; state <= S_ROPE_Q;
                                end
                                OP_O: begin index <= 0; state <= S_RESID_ATTN; end
                                OP_GATE: begin_gemv(OP_UP, up_base(layer_index), D_FF, D_MODEL);
                                OP_UP: begin index <= 0; state <= S_SWIGLU; end
                                OP_DOWN: begin index <= 0; state <= S_RESID_FF; end
                                default: state <= S_ARGMAX;
                            endcase
                        end else row <= row + 1;
                    end else begin
                        column <= column + 1;
                        mac_acc <= next_acc;
                    end
                end

                S_ROPE_Q: begin : rope_q_step
                    reg signed [31:0] c, s, odd_value;
                    integer vi;
                    vi = head*HEAD_DIM + pair_index*2;
                    c = parameter_memory[ROPE_COS_BASE + position*PAIRS_PER_POS + pair_index];
                    s = parameter_memory[ROPE_SIN_BASE + position*PAIRS_PER_POS + pair_index];
                    odd_value = q[vi+1];
                    q[vi] <= qadd(qmul(q[vi],c), -qmul(odd_value,s));
                    q[vi+1] <= qadd(qmul(q[vi],s), qmul(odd_value,c));
                    if (pair_index == PAIRS_PER_POS-1) begin
                        pair_index <= 0;
                        if (head == HEADS-1) begin head <= 0; state <= S_ROPE_K; end
                        else head <= head + 1;
                    end else pair_index <= pair_index + 1;
                end

                S_ROPE_K: begin : rope_k_step
                    reg signed [31:0] c, s, odd_value;
                    integer vi;
                    vi = head*HEAD_DIM + pair_index*2;
                    c = parameter_memory[ROPE_COS_BASE + position*PAIRS_PER_POS + pair_index];
                    s = parameter_memory[ROPE_SIN_BASE + position*PAIRS_PER_POS + pair_index];
                    odd_value = k[vi+1];
                    k[vi] <= qadd(qmul(k[vi],c), -qmul(odd_value,s));
                    k[vi+1] <= qadd(qmul(k[vi],s), qmul(odd_value,c));
                    if (pair_index == PAIRS_PER_POS-1) begin
                        pair_index <= 0;
                        if (head == KV_HEADS-1) begin index <= 0; state <= S_KV_STORE; end
                        else head <= head + 1;
                    end else pair_index <= pair_index + 1;
                end

                S_KV_STORE: begin
                    kv_key[(layer_index*MAX_CONTEXT + position)*KV_DIM + index] <= k[index];
                    kv_value[(layer_index*MAX_CONTEXT + position)*KV_DIM + index] <= v[index];
                    if (index == KV_DIM-1) begin
                        head <= 0; time_index <= 0; column <= 0; mac_acc <= 0;
                        state <= S_SCORE_INIT;
                    end else index <= index + 1;
                end

                S_SCORE_INIT: begin
                    column <= 0; mac_acc <= 0; state <= S_SCORE_DOT;
                end

                S_SCORE_DOT: begin : score_step
                    reg signed [31:0] next_score;
                    integer q_index, kv_index;
                    q_index = head*HEAD_DIM + column;
                    kv_index = ((layer_index*MAX_CONTEXT + time_index)*KV_DIM) +
                               (head/GROUPS)*HEAD_DIM + column;
                    next_score = qadd(mac_acc, qmul(q[q_index],kv_key[kv_index]));
                    if (column == HEAD_DIM-1) begin
                        scores[head*MAX_CONTEXT+time_index] <=
                            qmul(next_score, parameter_memory[ATTN_SCALE_ADDR]);
                        column <= 0; mac_acc <= 0;
                        if (time_index == position) begin
                            time_index <= 0; state <= S_SOFTMAX_MAX;
                            softmax_max <= -32'sh7fff_ffff;
                        end else time_index <= time_index + 1;
                    end else begin column <= column + 1; mac_acc <= next_score; end
                end

                S_SOFTMAX_MAX: begin
                    if (scores[head*MAX_CONTEXT+time_index] > softmax_max)
                        softmax_max <= scores[head*MAX_CONTEXT+time_index];
                    if (time_index == position) begin
                        // Include current value when it is the maximum.
                        if (scores[head*MAX_CONTEXT+time_index] > softmax_max)
                            softmax_max <= scores[head*MAX_CONTEXT+time_index];
                        time_index <= 0; softmax_sum <= 0; state <= S_SOFTMAX_EXP;
                    end else time_index <= time_index + 1;
                end

                S_SOFTMAX_EXP: begin : exp_step
                    reg signed [31:0] ev;
                    ev = qexp(qadd(scores[head*MAX_CONTEXT+time_index], -softmax_max));
                    probabilities[head*MAX_CONTEXT+time_index] <= ev;
                    softmax_sum <= qadd(softmax_sum, ev);
                    if (time_index == position) begin time_index <= 0; state <= S_SOFTMAX_DIV; end
                    else time_index <= time_index + 1;
                end

                S_SOFTMAX_DIV: begin
                    probabilities[head*MAX_CONTEXT+time_index] <=
                        qdiv(probabilities[head*MAX_CONTEXT+time_index], softmax_sum);
                    if (time_index == position) begin
                        time_index <= 0;
                        if (head == HEADS-1) begin head <= 0; column <= 0; state <= S_ATTN_INIT; end
                        else begin head <= head + 1; state <= S_SCORE_INIT; end
                    end else time_index <= time_index + 1;
                end

                S_ATTN_INIT: begin
                    time_index <= 0; mac_acc <= 0; state <= S_ATTN_SUM;
                end

                S_ATTN_SUM: begin : attention_step
                    reg signed [31:0] next_attention;
                    integer value_addr;
                    value_addr = ((layer_index*MAX_CONTEXT + time_index)*KV_DIM) +
                                 (head/GROUPS)*HEAD_DIM + column;
                    next_attention = qadd(mac_acc,
                        qmul(probabilities[head*MAX_CONTEXT+time_index],kv_value[value_addr]));
                    if (time_index == position) begin
                        attention[head*HEAD_DIM+column] <= next_attention;
                        time_index <= 0; mac_acc <= 0;
                        if (column == HEAD_DIM-1) begin
                            column <= 0;
                            if (head == HEADS-1)
                                begin_gemv(OP_O,o_base(layer_index),D_MODEL,D_MODEL);
                            else begin head <= head + 1; state <= S_ATTN_INIT; end
                        end else begin column <= column + 1; state <= S_ATTN_INIT; end
                    end else begin time_index <= time_index + 1; mac_acc <= next_attention; end
                end

                S_RESID_ATTN: begin
                    x[index] <= qadd(x[index], projected[index]);
                    if (index == D_MODEL-1) begin
                        index <= 0; norm_kind <= NORM_FFN; norm_sum <= 0; state <= S_NORM_SQ;
                    end else index <= index + 1;
                end

                S_SWIGLU: begin
                    ff[index] <= qmul(qsilu(gate[index]),up[index]);
                    if (index == D_FF-1)
                        begin_gemv(OP_DOWN,down_base(layer_index),D_MODEL,D_FF);
                    else index <= index + 1;
                end

                S_RESID_FF: begin
                    x[index] <= qadd(x[index], projected[index]);
                    if (index == D_MODEL-1) begin
                        index <= 0;
                        if (layer_index == LAYERS-1) begin
                            norm_kind <= NORM_FINAL; norm_sum <= 0; state <= S_NORM_SQ;
                        end else begin
                            layer_index <= layer_index + 1;
                            norm_kind <= NORM_ATTN; norm_sum <= 0; state <= S_NORM_SQ;
                        end
                    end else index <= index + 1;
                end

                S_ARGMAX: begin
                    output_token <= best_token;
                    state <= S_FINISH;
                end

                S_FINISH: begin
                    busy <= 0; done <= 1; position <= position + 1'b1; state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    initial begin
        if ((D_MODEL % HEADS) != 0 || (HEADS % KV_HEADS) != 0 ||
            (HEAD_DIM % 2) != 0 || HEAD_DIM < 2)
            $error("invalid Llama head dimensions");
    end
endmodule

`default_nettype wire
