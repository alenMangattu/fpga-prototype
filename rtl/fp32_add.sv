`timescale 1ns/1ps
`default_nettype none

// Combinational IEEE-754 binary32 adder with round-to-nearest-even.
module fp32_add (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] result
);
    logic sign_a, sign_b, sign_big, sign_small, sign_result;
    logic [7:0] exp_a_field, exp_b_field;
    logic [22:0] frac_a, frac_b;
    logic [23:0] sig_a, sig_b, sig_big, sig_small;
    logic [26:0] ext_big, ext_small, shifted_small, ext_result;
    logic [27:0] add_result;
    logic [24:0] rounded;
    logic sticky_bit, round_up;
    integer exp_a, exp_b, exp_big, exp_result, shift_count, i;

    // Use always @* for compatibility with Icarus, which emits misleading
    // "constant selects in always_*" notices for an always_comb block.
    always @* begin
        sign_a = a[31];
        sign_b = b[31];
        exp_a_field = a[30:23];
        exp_b_field = b[30:23];
        frac_a = a[22:0];
        frac_b = b[22:0];
        sig_a = (exp_a_field == 0) ? {1'b0, frac_a} : {1'b1, frac_a};
        sig_b = (exp_b_field == 0) ? {1'b0, frac_b} : {1'b1, frac_b};
        if (exp_a_field == 0)
            exp_a = 1;
        else
            exp_a = {24'd0, exp_a_field};
        if (exp_b_field == 0)
            exp_b = 1;
        else
            exp_b = {24'd0, exp_b_field};
        sign_big = 0;
        sign_small = 0;
        sign_result = 0;
        sig_big = 0;
        sig_small = 0;
        exp_big = 0;
        exp_result = 0;
        ext_big = 0;
        ext_small = 0;
        shifted_small = 0;
        ext_result = 0;
        add_result = 0;
        rounded = 0;
        sticky_bit = 0;
        round_up = 0;
        shift_count = 0;
        result = 0;

        if (((exp_a_field == 8'hff) && (frac_a != 0)) ||
            ((exp_b_field == 8'hff) && (frac_b != 0)) ||
            ((exp_a_field == 8'hff) && (exp_b_field == 8'hff) &&
             (frac_a == 0) && (frac_b == 0) && (sign_a != sign_b))) begin
            result = 32'h7fc0_0000;
        end else if (exp_a_field == 8'hff) begin
            result = {sign_a, 8'hff, 23'd0};
        end else if (exp_b_field == 8'hff) begin
            result = {sign_b, 8'hff, 23'd0};
        end else if ((exp_a_field == 0) && (frac_a == 0) &&
                     (exp_b_field == 0) && (frac_b == 0)) begin
            result = {(sign_a & sign_b), 31'd0};
        end else begin
            if ((exp_a > exp_b) || ((exp_a == exp_b) && (sig_a >= sig_b))) begin
                sign_big = sign_a;
                sign_small = sign_b;
                sig_big = sig_a;
                sig_small = sig_b;
                exp_big = exp_a;
                shift_count = exp_a - exp_b;
            end else begin
                sign_big = sign_b;
                sign_small = sign_a;
                sig_big = sig_b;
                sig_small = sig_a;
                exp_big = exp_b;
                shift_count = exp_b - exp_a;
            end

            ext_big = {sig_big, 3'b000};
            ext_small = {sig_small, 3'b000};
            if (shift_count >= 27) begin
                shifted_small = (ext_small != 0) ? 27'd1 : 27'd0;
            end else begin
                shifted_small = ext_small >> shift_count;
                for (i = 0; i < 27; i = i + 1)
                    if (i < shift_count)
                        sticky_bit = sticky_bit | ext_small[i];
                shifted_small[0] = shifted_small[0] | sticky_bit;
            end

            exp_result = exp_big;
            sign_result = sign_big;
            if (sign_big == sign_small) begin
                add_result = {1'b0, ext_big} + {1'b0, shifted_small};
                if (add_result[27]) begin
                    ext_result = add_result[27:1];
                    ext_result[0] = ext_result[0] | add_result[0];
                    exp_result = exp_result + 1;
                end else begin
                    ext_result = add_result[26:0];
                end
            end else begin
                ext_result = ext_big - shifted_small;
                if (ext_result == 0) begin
                    sign_result = 0;
                end else begin
                    for (i = 0; i < 26; i = i + 1)
                        if (!ext_result[26] && (exp_result > 1)) begin
                            ext_result = ext_result << 1;
                            exp_result = exp_result - 1;
                        end
                end
            end

            if (ext_result == 0) begin
                result = {sign_result, 31'd0};
            end else if (exp_result >= 255) begin
                result = {sign_result, 8'hff, 23'd0};
            end else begin
                round_up = ext_result[2] &
                           (ext_result[1] | ext_result[0] | ext_result[3]);
                rounded = {1'b0, ext_result[26:3]} + round_up;
                if (rounded[24]) begin
                    rounded = rounded >> 1;
                    exp_result = exp_result + 1;
                end

                if (exp_result >= 255)
                    result = {sign_result, 8'hff, 23'd0};
                else if ((exp_result == 1) && !rounded[23])
                    result = {sign_result, 8'h00, rounded[22:0]};
                else
                    result = {sign_result, exp_result[7:0], rounded[22:0]};
            end
        end
        i = 0;
    end
endmodule

`default_nettype wire
