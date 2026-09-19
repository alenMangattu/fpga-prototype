`timescale 1ns/1ps
`default_nettype none

module fp16_to_fp32 (
    input  logic [15:0] value,
    output logic [31:0] result
);
    logic sign;
    logic [4:0] exponent;
    logic [9:0] fraction;
    logic [10:0] significand;
    integer unbiased;
    integer shifts;

    always @* begin
        sign = value[15];
        exponent = value[14:10];
        fraction = value[9:0];
        significand = {1'b0, fraction};
        unbiased = 0;
        shifts = 0;
        result = 32'd0;

        if (exponent == 5'h1f) begin
            result = {sign, 8'hff, fraction, 13'd0};
            if (fraction != 0)
                result[22] = 1'b1;
        end else if (exponent != 0) begin
            result = {sign, 8'(exponent + 112), fraction, 13'd0};
        end else if (fraction != 0) begin
            unbiased = -14;
            for (shifts = 0; shifts < 10; shifts = shifts + 1)
                if (!significand[10]) begin
                    significand = significand << 1;
                    unbiased = unbiased - 1;
                end
            result = {sign, 8'(unbiased + 127), significand[9:0], 13'd0};
        end else begin
            result = {sign, 31'd0};
        end
    end
endmodule

`default_nettype wire
