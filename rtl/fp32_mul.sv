`timescale 1ns/1ps
`default_nettype none

// Combinational IEEE-754 binary32 multiplier with round-to-nearest-even.
module fp32_mul (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] result
);
    logic sign_result;
    logic [7:0] exp_a_field, exp_b_field;
    logic [22:0] frac_a, frac_b;
    logic [23:0] sig_a, sig_b;
    logic [47:0] product, normalized;
    logic [23:0] truncated;
    logic [24:0] rounded;
    logic guard_bit, sticky_bit, round_up;
    integer exp_a, exp_b, exp_result, shift_count, i;

    // Use always @* for compatibility with Icarus, which emits misleading
    // "constant selects in always_*" notices for an always_comb block.
    always @* begin
        sign_result = a[31] ^ b[31];
        exp_a_field = a[30:23];
        exp_b_field = b[30:23];
        frac_a = a[22:0];
        frac_b = b[22:0];
        sig_a = 0;
        sig_b = 0;
        product = 0;
        normalized = 0;
        truncated = 0;
        rounded = 0;
        guard_bit = 0;
        sticky_bit = 0;
        round_up = 0;
        exp_a = 0;
        exp_b = 0;
        exp_result = 0;
        shift_count = 0;
        result = 0;

        if (((exp_a_field == 8'hff) && (frac_a != 0)) ||
            ((exp_b_field == 8'hff) && (frac_b != 0)) ||
            (((exp_a_field == 8'hff) && (frac_a == 0)) &&
             ((exp_b_field == 0) && (frac_b == 0))) ||
            (((exp_b_field == 8'hff) && (frac_b == 0)) &&
             ((exp_a_field == 0) && (frac_a == 0)))) begin
            result = 32'h7fc0_0000;
        end else if ((exp_a_field == 8'hff) || (exp_b_field == 8'hff)) begin
            result = {sign_result, 8'hff, 23'd0};
        end else if (((exp_a_field == 0) && (frac_a == 0)) ||
                     ((exp_b_field == 0) && (frac_b == 0))) begin
            result = {sign_result, 31'd0};
        end else begin
            if (exp_a_field == 0) begin
                sig_a = {1'b0, frac_a};
                exp_a = -126;
                for (i = 0; i < 23; i = i + 1)
                    if (!sig_a[23]) begin
                        sig_a = sig_a << 1;
                        exp_a = exp_a - 1;
                    end
            end else begin
                sig_a = {1'b1, frac_a};
                exp_a = {24'd0, exp_a_field};
                exp_a = exp_a - 127;
            end

            if (exp_b_field == 0) begin
                sig_b = {1'b0, frac_b};
                exp_b = -126;
                for (i = 0; i < 23; i = i + 1)
                    if (!sig_b[23]) begin
                        sig_b = sig_b << 1;
                        exp_b = exp_b - 1;
                    end
            end else begin
                sig_b = {1'b1, frac_b};
                exp_b = {24'd0, exp_b_field};
                exp_b = exp_b - 127;
            end

            product = sig_a * sig_b;
            exp_result = exp_a + exp_b;
            if (product[47]) begin
                normalized = product >> 1;
                normalized[0] = normalized[0] | product[0];
                exp_result = exp_result + 1;
            end else begin
                normalized = product;
            end

            if (exp_result > 127) begin
                result = {sign_result, 8'hff, 23'd0};
            end else begin
                if (exp_result >= -126)
                    shift_count = 23;
                else
                    shift_count = 23 + (-126 - exp_result);

                truncated = (shift_count >= 48) ? 24'd0 :
                            24'(normalized >> shift_count);
                if ((shift_count > 0) && (shift_count <= 48))
                    guard_bit = normalized[shift_count - 1];
                for (i = 0; i < 48; i = i + 1)
                    if (i < (shift_count - 1))
                        sticky_bit = sticky_bit | normalized[i];

                round_up = guard_bit & (sticky_bit | truncated[0]);
                rounded = {1'b0, truncated[23:0]} + round_up;

                if (exp_result >= -126) begin
                    if (rounded[24]) begin
                        rounded = rounded >> 1;
                        exp_result = exp_result + 1;
                    end
                    if (exp_result > 127)
                        result = {sign_result, 8'hff, 23'd0};
                    else
                        result = {sign_result, 8'(exp_result + 127),
                                  rounded[22:0]};
                end else if (rounded[23]) begin
                    result = {sign_result, 8'h01, 23'd0};
                end else begin
                    result = {sign_result, 8'h00, rounded[22:0]};
                end
            end
        end
        i = 0;
    end
endmodule

`default_nettype wire
