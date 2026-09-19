`timescale 1ns/1ps
`default_nettype none

// Bit-rounded FP32 reference for an exact GPT-2 pre-LayerNorm block:
//   r1 = x + Attention(LayerNorm(x))
//   y  = r1 + MLP_GELU(LayerNorm(r1))
//
// This module is an executable golden model for verification. It uses
// SystemVerilog shortreal math for sqrt/exp/tanh and is NOT synthesizable.
// The production controller must bind these operations to pipelined FPGA IP.
module gpt2_block_exact_sim #(
    parameter integer SEQ_LEN = 2,
    parameter integer D_MODEL = 4,
    parameter integer D_FF = 8,
    parameter logic [31:0] LN_EPSILON = 32'h3727_c5ac // 1.0e-5
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic [SEQ_LEN*D_MODEL*32-1:0] x,
    input  logic [D_MODEL*32-1:0] ln1_gamma,
    input  logic [D_MODEL*32-1:0] ln1_beta,
    input  logic [D_MODEL*D_MODEL*32-1:0] weight_q,
    input  logic [D_MODEL*32-1:0] bias_q,
    input  logic [D_MODEL*D_MODEL*32-1:0] weight_k,
    input  logic [D_MODEL*32-1:0] bias_k,
    input  logic [D_MODEL*D_MODEL*32-1:0] weight_v,
    input  logic [D_MODEL*32-1:0] bias_v,
    input  logic [D_MODEL*D_MODEL*32-1:0] weight_o,
    input  logic [D_MODEL*32-1:0] bias_o,
    input  logic [D_MODEL*32-1:0] ln2_gamma,
    input  logic [D_MODEL*32-1:0] ln2_beta,
    input  logic [D_FF*D_MODEL*32-1:0] weight_ff1,
    input  logic [D_FF*32-1:0] bias_ff1,
    input  logic [D_MODEL*D_FF*32-1:0] weight_ff2,
    input  logic [D_MODEL*32-1:0] bias_ff2,
    output logic [SEQ_LEN*D_MODEL*32-1:0] y,
    output logic done
);
    real xr [0:SEQ_LEN-1][0:D_MODEL-1];
    real n1 [0:SEQ_LEN-1][0:D_MODEL-1];
    real q [0:SEQ_LEN-1][0:D_MODEL-1];
    real k [0:SEQ_LEN-1][0:D_MODEL-1];
    real v [0:SEQ_LEN-1][0:D_MODEL-1];
    real score [0:SEQ_LEN-1][0:SEQ_LEN-1];
    real probability [0:SEQ_LEN-1][0:SEQ_LEN-1];
    real attention_value [0:SEQ_LEN-1][0:D_MODEL-1];
    real residual1 [0:SEQ_LEN-1][0:D_MODEL-1];
    real n2 [0:SEQ_LEN-1][0:D_MODEL-1];
    real ff [0:SEQ_LEN-1][0:D_FF-1];
    real yr [0:SEQ_LEN-1][0:D_MODEL-1];
    real accumulator, mean, variance, delta, inverse_std;
    real maximum_score, exponential_sum, scale, epsilon;
    integer token, row, column, key_token;

    function automatic real from_bits(input logic [31:0] bits);
        shortreal value;
        begin
            value = $bitstoshortreal(bits);
            from_bits = value;
        end
    endfunction

    function automatic logic [31:0] to_bits(input real value);
        shortreal rounded;
        begin
            rounded = value;
            to_bits = $shortrealtobits(rounded);
        end
    endfunction

    function automatic real round_fp32(input real value);
        begin
            round_fp32 = from_bits(to_bits(value));
        end
    endfunction

    function automatic real fp_add(input real a, input real b);
        fp_add = round_fp32(a + b);
    endfunction
    function automatic real fp_mul(input real a, input real b);
        fp_mul = round_fp32(a * b);
    endfunction
    function automatic real fp_div(input real a, input real b);
        fp_div = round_fp32(a / b);
    endfunction
    function automatic real fp_sqrt(input real a);
        fp_sqrt = round_fp32($sqrt(a));
    endfunction
    function automatic real fp_exp(input real a);
        fp_exp = round_fp32($exp(a));
    endfunction
    function automatic real fp_tanh(input real a);
        fp_tanh = round_fp32($tanh(a));
    endfunction

    function automatic real gpt2_gelu(input real value);
        real value2, value3, inner, tanh_value;
        begin
            value2 = fp_mul(value, value);
            value3 = fp_mul(value2, value);
            inner = fp_add(value, fp_mul(0.044715, value3));
            inner = fp_mul(0.7978845608028654, inner);
            tanh_value = fp_tanh(inner);
            gpt2_gelu = fp_mul(fp_mul(0.5, value),
                               fp_add(1.0, tanh_value));
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            y <= 0;
            done <= 0;
        end else begin
            done <= 0;
            if (start) begin
                epsilon = from_bits(LN_EPSILON);
                scale = fp_div(1.0, fp_sqrt(D_MODEL));

                for (token = 0; token < SEQ_LEN; token = token + 1)
                    for (column = 0; column < D_MODEL; column = column + 1)
                        xr[token][column] =
                            from_bits(x[(token*D_MODEL + column)*32 +: 32]);

                // First LayerNorm.
                for (token = 0; token < SEQ_LEN; token = token + 1) begin
                    mean = 0.0;
                    for (column = 0; column < D_MODEL; column = column + 1)
                        mean = fp_add(mean, xr[token][column]);
                    mean = fp_div(mean, D_MODEL);
                    variance = 0.0;
                    for (column = 0; column < D_MODEL; column = column + 1) begin
                        delta = fp_add(xr[token][column], -mean);
                        variance = fp_add(variance, fp_mul(delta, delta));
                    end
                    variance = fp_div(variance, D_MODEL);
                    inverse_std = fp_div(1.0, fp_sqrt(fp_add(variance, epsilon)));
                    for (column = 0; column < D_MODEL; column = column + 1) begin
                        delta = fp_add(xr[token][column], -mean);
                        n1[token][column] = fp_add(
                            fp_mul(fp_mul(delta, inverse_std),
                                   from_bits(ln1_gamma[column*32 +: 32])),
                            from_bits(ln1_beta[column*32 +: 32]));
                    end
                end

                // Q, K, and V projections.
                for (token = 0; token < SEQ_LEN; token = token + 1)
                    for (row = 0; row < D_MODEL; row = row + 1) begin
                        q[token][row] = from_bits(bias_q[row*32 +: 32]);
                        k[token][row] = from_bits(bias_k[row*32 +: 32]);
                        v[token][row] = from_bits(bias_v[row*32 +: 32]);
                        for (column = 0; column < D_MODEL; column = column + 1) begin
                            q[token][row] = fp_add(q[token][row], fp_mul(
                                n1[token][column],
                                from_bits(weight_q[(row*D_MODEL+column)*32 +: 32])));
                            k[token][row] = fp_add(k[token][row], fp_mul(
                                n1[token][column],
                                from_bits(weight_k[(row*D_MODEL+column)*32 +: 32])));
                            v[token][row] = fp_add(v[token][row], fp_mul(
                                n1[token][column],
                                from_bits(weight_v[(row*D_MODEL+column)*32 +: 32])));
                        end
                    end

                // Scaled causal softmax attention.
                for (token = 0; token < SEQ_LEN; token = token + 1) begin
                    maximum_score = -1.0e30;
                    for (key_token = 0; key_token < SEQ_LEN;
                         key_token = key_token + 1) begin
                        if (key_token <= token) begin
                            accumulator = 0.0;
                            for (column = 0; column < D_MODEL;
                                 column = column + 1)
                                accumulator = fp_add(accumulator,
                                    fp_mul(q[token][column], k[key_token][column]));
                            score[token][key_token] = fp_mul(accumulator, scale);
                            if (score[token][key_token] > maximum_score)
                                maximum_score = score[token][key_token];
                        end else begin
                            score[token][key_token] = -1.0e30;
                        end
                    end
                    exponential_sum = 0.0;
                    for (key_token = 0; key_token <= token;
                         key_token = key_token + 1) begin
                        probability[token][key_token] = fp_exp(
                            fp_add(score[token][key_token], -maximum_score));
                        exponential_sum = fp_add(exponential_sum,
                            probability[token][key_token]);
                    end
                    for (key_token = 0; key_token <= token;
                         key_token = key_token + 1)
                        probability[token][key_token] = fp_div(
                            probability[token][key_token], exponential_sum);

                    for (column = 0; column < D_MODEL; column = column + 1) begin
                        attention_value[token][column] = 0.0;
                        for (key_token = 0; key_token <= token;
                             key_token = key_token + 1)
                            attention_value[token][column] = fp_add(
                                attention_value[token][column], fp_mul(
                                    probability[token][key_token],
                                    v[key_token][column]));
                    end
                end

                // Attention output projection and first residual.
                for (token = 0; token < SEQ_LEN; token = token + 1)
                    for (row = 0; row < D_MODEL; row = row + 1) begin
                        accumulator = from_bits(bias_o[row*32 +: 32]);
                        for (column = 0; column < D_MODEL; column = column + 1)
                            accumulator = fp_add(accumulator, fp_mul(
                                attention_value[token][column],
                                from_bits(weight_o[(row*D_MODEL+column)*32 +: 32])));
                        residual1[token][row] = fp_add(xr[token][row], accumulator);
                    end

                // Second LayerNorm.
                for (token = 0; token < SEQ_LEN; token = token + 1) begin
                    mean = 0.0;
                    for (column = 0; column < D_MODEL; column = column + 1)
                        mean = fp_add(mean, residual1[token][column]);
                    mean = fp_div(mean, D_MODEL);
                    variance = 0.0;
                    for (column = 0; column < D_MODEL; column = column + 1) begin
                        delta = fp_add(residual1[token][column], -mean);
                        variance = fp_add(variance, fp_mul(delta, delta));
                    end
                    variance = fp_div(variance, D_MODEL);
                    inverse_std = fp_div(1.0, fp_sqrt(fp_add(variance, epsilon)));
                    for (column = 0; column < D_MODEL; column = column + 1) begin
                        delta = fp_add(residual1[token][column], -mean);
                        n2[token][column] = fp_add(
                            fp_mul(fp_mul(delta, inverse_std),
                                   from_bits(ln2_gamma[column*32 +: 32])),
                            from_bits(ln2_beta[column*32 +: 32]));
                    end
                end

                // GPT-2 MLP with tanh-form GELU and final residual.
                for (token = 0; token < SEQ_LEN; token = token + 1) begin
                    for (row = 0; row < D_FF; row = row + 1) begin
                        accumulator = from_bits(bias_ff1[row*32 +: 32]);
                        for (column = 0; column < D_MODEL; column = column + 1)
                            accumulator = fp_add(accumulator, fp_mul(
                                n2[token][column],
                                from_bits(weight_ff1[(row*D_MODEL+column)*32 +: 32])));
                        ff[token][row] = gpt2_gelu(accumulator);
                    end
                    for (row = 0; row < D_MODEL; row = row + 1) begin
                        accumulator = from_bits(bias_ff2[row*32 +: 32]);
                        for (column = 0; column < D_FF; column = column + 1)
                            accumulator = fp_add(accumulator, fp_mul(
                                ff[token][column],
                                from_bits(weight_ff2[(row*D_FF+column)*32 +: 32])));
                        yr[token][row] = fp_add(residual1[token][row], accumulator);
                        y[(token*D_MODEL + row)*32 +: 32] <=
                            to_bits(yr[token][row]);
                    end
                end
                done <= 1;
            end
        end
    end
endmodule

`default_nettype wire
