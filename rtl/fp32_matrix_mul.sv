`timescale 1ns/1ps
`default_nettype none

// Fully parallel FP32 matrix multiplication:
//
//     A[M x K] * B[K x N] = C[M x N]
//
// Matrices use row-major packing. Element (row, column) starts at bit
// ((row * column_count + column) * 32) of its bus. The implementation creates
// M*N dot-product engines, so the complete matrix is captured in one cycle.
module fp32_matrix_mul #(
    parameter integer M = 2,
    parameter integer K = 3,
    parameter integer N = 2
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  start,
    input  logic [M*K*32-1:0]     matrix_a,
    input  logic [K*N*32-1:0]     matrix_b,
    output logic [M*N*32-1:0]     matrix_c,
    output logic                  valid
);
    logic [31:0] cell_result [0:M*N-1];
    logic        cell_valid  [0:M*N-1];

    genvar row;
    genvar column;
    genvar inner;
    generate
        for (row = 0; row < M; row = row + 1) begin : output_row
            for (column = 0; column < N; column = column + 1) begin : output_column
                logic [K*32-1:0] row_vector;
                logic [K*32-1:0] column_vector;

                for (inner = 0; inner < K; inner = inner + 1) begin : gather
                    assign row_vector[inner*32 +: 32] =
                        matrix_a[(row*K + inner)*32 +: 32];
                    assign column_vector[inner*32 +: 32] =
                        matrix_b[(inner*N + column)*32 +: 32];
                end

                fp32_dot_product #(.LENGTH(K)) dot_product (
                    .clk      (clk),
                    .rst      (rst),
                    .start    (start),
                    .vector_a (row_vector),
                    .vector_b (column_vector),
                    .result   (cell_result[row*N + column]),
                    .valid    (cell_valid[row*N + column])
                );

                assign matrix_c[(row*N + column)*32 +: 32] =
                    cell_result[row*N + column];
            end
        end
    endgenerate

    assign valid = cell_valid[0];
endmodule

`default_nettype wire
