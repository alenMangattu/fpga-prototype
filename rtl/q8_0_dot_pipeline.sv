`timescale 1ns/1ps
`default_nettype none

// 32 signed 8x8 multipliers followed by a registered balanced reduction tree.
// Accepts one block per clock. Six register stages, with tags/scales following
// valid through bubbles. Products and sums are widened before every addition.
module q8_0_dot_pipeline #(
    parameter integer TAG_BITS = 1
) (
    input wire clk, rst, input_valid,
    input wire [255:0] weights, activations,
    input wire [15:0] weight_scale, activation_scale,
    input wire [TAG_BITS-1:0] input_tag,
    output wire output_valid,
    output wire signed [31:0] integer_dot,
    output wire [15:0] output_weight_scale, output_activation_scale,
    output wire [TAG_BITS-1:0] output_tag
);
    reg signed [15:0] products [0:31];
    reg signed [16:0] sum16 [0:15];
    reg signed [17:0] sum8 [0:7];
    reg signed [18:0] sum4 [0:3];
    reg signed [19:0] sum2 [0:1];
    reg signed [20:0] sum1;
    reg [5:0] valid;
    reg [15:0] ws [0:5], xs [0:5];
    reg [TAG_BITS-1:0] tags [0:5];
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            valid <= 0;
        end else begin
            valid <= {valid[4:0], input_valid};
            for (i=0; i<32; i=i+1)
                products[i] <= $signed(weights[i*8 +: 8]) * $signed(activations[i*8 +: 8]);
            for (i=0; i<16; i=i+1)
                sum16[i] <= {products[2*i][15], products[2*i]} + {products[2*i+1][15], products[2*i+1]};
            for (i=0; i<8; i=i+1)
                sum8[i] <= {sum16[2*i][16], sum16[2*i]} + {sum16[2*i+1][16], sum16[2*i+1]};
            for (i=0; i<4; i=i+1)
                sum4[i] <= {sum8[2*i][17], sum8[2*i]} + {sum8[2*i+1][17], sum8[2*i+1]};
            for (i=0; i<2; i=i+1)
                sum2[i] <= {sum4[2*i][18], sum4[2*i]} + {sum4[2*i+1][18], sum4[2*i+1]};
            sum1 <= {sum2[0][19], sum2[0]} + {sum2[1][19], sum2[1]};
            ws[0] <= weight_scale;
            xs[0] <= activation_scale;
            tags[0] <= input_tag;
            for (i=1; i<6; i=i+1) begin
                ws[i] <= ws[i-1]; xs[i] <= xs[i-1]; tags[i] <= tags[i-1];
            end
        end
    end
    assign output_valid = valid[5];
    assign integer_dot = {{11{sum1[20]}}, sum1};
    assign output_weight_scale = ws[5];
    assign output_activation_scale = xs[5];
    assign output_tag = tags[5];
endmodule
`default_nettype wire
