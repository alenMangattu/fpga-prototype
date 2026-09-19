`timescale 1ns/1ps
`default_nettype none

// FP32 ReLU: max(0, value). NaNs are propagated unchanged; negative values
// and negative zero become positive zero.
module fp32_relu (
    input  logic [31:0] value,
    output logic [31:0] result
);
    always @* begin
        if ((value[30:23] == 8'hff) && (value[22:0] != 0))
            result = value;
        else if (value[31])
            result = 32'h0000_0000;
        else
            result = value;
    end
endmodule

`default_nettype wire
