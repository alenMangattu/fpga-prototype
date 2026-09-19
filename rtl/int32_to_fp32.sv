`timescale 1ns/1ps
`default_nettype none

// Signed integer to binary32, round-to-nearest-even.
module int32_to_fp32 (
    input  logic signed [31:0] value,
    output logic [31:0] result
);
    logic sign;
    logic [32:0] magnitude;
    logic [32:0] shifted;
    logic [24:0] rounded;
    logic guard_bit, sticky_bit;
    integer highest, shift, i;

    always @* begin
        sign = value[31];
        magnitude = sign ? (33'(-value)) : {1'b0, value};
        shifted = 0;
        rounded = 0;
        guard_bit = 0;
        sticky_bit = 0;
        highest = 0;
        shift = 0;
        result = 0;

        for (i = 0; i < 32; i = i + 1)
            if (magnitude[i])
                highest = i;

        if (magnitude == 0) begin
            result = 0;
        end else if (highest <= 23) begin
            shifted = magnitude << (23 - highest);
            result = {sign, 8'(highest + 127), shifted[22:0]};
        end else begin
            shift = highest - 23;
            shifted = magnitude >> shift;
            guard_bit = magnitude[shift - 1];
            for (i = 0; i < 32; i = i + 1)
                if (i < shift - 1)
                    sticky_bit = sticky_bit | magnitude[i];
            rounded = {1'b0, shifted[23:0]}
                    + (guard_bit & (sticky_bit | shifted[0]));
            if (rounded[24])
                result = {sign, 8'(highest + 128), 23'd0};
            else
                result = {sign, 8'(highest + 127), rounded[22:0]};
        end
        i = 0;
    end
endmodule

`default_nettype wire
