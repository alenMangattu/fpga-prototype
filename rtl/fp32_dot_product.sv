`timescale 1ns/1ps
`default_nettype none

// Parameterized, fully parallel FP32 dot product. LENGTH need not be a power
// of two; unused leaves in the balanced tree are automatically padded with 0.
module fp32_dot_product #(
    parameter integer LENGTH = 4
) (
    input  logic                   clk,
    input  logic                   rst,
    input  logic                   start,
    input  logic [LENGTH*32-1:0]   vector_a,
    input  logic [LENGTH*32-1:0]   vector_b,
    output logic [31:0]            result,
    output logic                   valid
);
    localparam integer TREE_SIZE = 1 << $clog2(LENGTH);
    localparam integer NODE_COUNT = (2 * TREE_SIZE) - 1;

    // Heap-style binary tree: node 0 is the root; children of node i are
    // 2*i+1 and 2*i+2; leaves begin at TREE_SIZE-1.
    logic [31:0] tree [0:NODE_COUNT-1];

    genvar lane;
    generate
        for (lane = 0; lane < TREE_SIZE; lane = lane + 1) begin : multiply
            if (lane < LENGTH) begin : active_lane
                fp32_mul multiplier (
                    .a      (vector_a[lane*32 +: 32]),
                    .b      (vector_b[lane*32 +: 32]),
                    .result (tree[TREE_SIZE - 1 + lane])
                );
            end else begin : padding_lane
                assign tree[TREE_SIZE - 1 + lane] = 32'h0000_0000;
            end
        end
    endgenerate

    genvar node;
    generate
        for (node = 0; node < TREE_SIZE - 1; node = node + 1) begin : add_node
            fp32_add adder (
                .a      (tree[node*2 + 1]),
                .b      (tree[node*2 + 2]),
                .result (tree[node])
            );
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            result <= 32'h0000_0000;
            valid <= 1'b0;
        end else begin
            valid <= start;
            if (start)
                result <= tree[0];
        end
    end
endmodule

`default_nettype wire
