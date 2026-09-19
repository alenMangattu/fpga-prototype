`timescale 1ns/1ps
`default_nettype none

// Fully parallel 16-element FP32 dot-product engine.
//
// Element i occupies bits [i*32 +: 32] of each input bus. All sixteen
// products and the balanced addition tree are combinational. When start is
// high at a rising edge, their result is captured and valid pulses high.
module fp32_dot16 (
    input  logic         clk,
    input  logic         rst,
    input  logic         start,
    input  logic [511:0] vector_a,
    input  logic [511:0] vector_b,
    output logic [31:0]  result,
    output logic         valid
);
    logic [31:0] product [0:15];
    logic [31:0] sum_l1  [0:7];
    logic [31:0] sum_l2  [0:3];
    logic [31:0] sum_l3  [0:1];
    logic [31:0] dot_product;

    genvar i;
    generate
        // Level 0: 16 multiplications happen at the same time.
        for (i = 0; i < 16; i = i + 1) begin : generate_multipliers
            fp32_mul multiplier (
                .a      (vector_a[i*32 +: 32]),
                .b      (vector_b[i*32 +: 32]),
                .result (product[i])
            );
        end

        // Level 1: 16 products become 8 partial sums.
        for (i = 0; i < 8; i = i + 1) begin : generate_level_1
            fp32_add adder (
                .a      (product[i*2]),
                .b      (product[i*2 + 1]),
                .result (sum_l1[i])
            );
        end

        // Level 2: 8 partial sums become 4.
        for (i = 0; i < 4; i = i + 1) begin : generate_level_2
            fp32_add adder (
                .a      (sum_l1[i*2]),
                .b      (sum_l1[i*2 + 1]),
                .result (sum_l2[i])
            );
        end

        // Level 3: 4 partial sums become 2.
        for (i = 0; i < 2; i = i + 1) begin : generate_level_3
            fp32_add adder (
                .a      (sum_l2[i*2]),
                .b      (sum_l2[i*2 + 1]),
                .result (sum_l3[i])
            );
        end
    endgenerate

    // Level 4: the final adder produces the complete dot product.
    fp32_add final_adder (
        .a      (sum_l3[0]),
        .b      (sum_l3[1]),
        .result (dot_product)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            result <= 32'h0000_0000;
            valid <= 1'b0;
        end else begin
            valid <= start;
            if (start)
                result <= dot_product;
        end
    end
endmodule

`default_nettype wire
