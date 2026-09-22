`timescale 1ns/1ps
`default_nettype none
`include "tile_config.svh"

// Fixed, trained Llama matrix tile. Only activations are writable.
// Generated llama_tile_rom contains literal model constants; synthesis can
// absorb them into gates. This is not a physical ROM macro or a full decoder.
// The result stream has no backpressure: consume every result_valid pulse.
module llama_frozen_tile_top (
    input wire clk, reset_n, start,
    input wire act_we,
    input wire [31:0] act_address,
    input wire [271:0] act_data,
    output wire start_ready,
    output reg busy, done,
    output reg result_valid,
    output reg [31:0] result_row, result_fp32, cycle_count
);
    localparam integer COLUMNS = `LLAMA_TILE_COLUMNS;
    localparam integer ROWS = `LLAMA_TILE_ROWS;
    localparam integer BLOCKS = COLUMNS / 32;
    localparam integer TOTAL = ROWS * BLOCKS;
    localparam integer ADDR_BITS = (TOTAL > 1) ? $clog2(TOTAL) : 1;
    reg [271:0] activation_memory [0:BLOCKS-1];
    reg [BLOCKS-1:0] loaded;
    reg issuing;
    reg [ADDR_BITS-1:0] address;
    wire [271:0] weight_word;
    wire [271:0] activation_word = activation_memory[address % BLOCKS];
    wire mac_valid;
    wire signed [31:0] dot;
    wire [15:0] ws, xs;
    wire [ADDR_BITS-1:0] mac_tag;
    wire [31:0] dot_float, ws_float, xs_float, weighted, scaled;
    reg scaled_valid;
    reg [31:0] scaled_value;
    reg [ADDR_BITS-1:0] scaled_tag;
    reg [31:0] accumulator;
    wire [31:0] addend = (scaled_tag % BLOCKS == 0) ? 32'd0 : accumulator;
    wire [31:0] accumulated;
    assign start_ready = !busy && (&loaded) && !act_we;

    llama_tile_rom rom(.address(address), .data(weight_word));
    q8_0_dot_pipeline #(.TAG_BITS(ADDR_BITS)) mac(
        .clk(clk), .rst(!reset_n), .input_valid(issuing),
        .weights(weight_word[271:16]), .activations(activation_word[271:16]),
        .weight_scale(weight_word[15:0]), .activation_scale(activation_word[15:0]),
        .input_tag(address), .output_valid(mac_valid), .integer_dot(dot),
        .output_weight_scale(ws), .output_activation_scale(xs), .output_tag(mac_tag));
    int32_to_fp32 convert_dot(.value(dot), .result(dot_float));
    fp16_to_fp32 convert_ws(.value(ws), .result(ws_float));
    fp16_to_fp32 convert_xs(.value(xs), .result(xs_float));
    fp32_mul scale_weight(.a(dot_float), .b(ws_float), .result(weighted));
    fp32_mul scale_activation(.a(weighted), .b(xs_float), .result(scaled));
    fp32_add accumulate(.a(addend), .b(scaled_value), .result(accumulated));

    always @(posedge clk) begin
        if (!reset_n) begin
            loaded <= 0; issuing <= 0; address <= 0;
            busy <= 0; done <= 0; result_valid <= 0;
            result_row <= 0; result_fp32 <= 0; cycle_count <= 0;
            scaled_valid <= 0; scaled_value <= 0; scaled_tag <= 0;
            accumulator <= 0;
        end else begin
            done <= 0;
            result_valid <= 0;
            scaled_valid <= mac_valid;
            if (mac_valid) begin
                scaled_value <= scaled;
                scaled_tag <= mac_tag;
            end
            if (act_we && !busy && act_address < BLOCKS) begin
                activation_memory[act_address] <= act_data;
                loaded[act_address] <= 1'b1;
            end
            if (start && start_ready) begin
                issuing <= 1; address <= 0; busy <= 1; cycle_count <= 0;
            end
            if (busy) cycle_count <= cycle_count + 1;
            if (issuing) begin
                if (address == TOTAL-1) issuing <= 0;
                else address <= address + 1'b1;
            end
            if (scaled_valid && busy) begin
                accumulator <= accumulated;
                if (scaled_tag % BLOCKS == BLOCKS-1) begin
                    result_valid <= 1;
                    result_row <= scaled_tag / BLOCKS;
                    result_fp32 <= accumulated;
                end
                if (scaled_tag == TOTAL-1) begin
                    done <= 1; busy <= 0;
                end
            end
        end
    end
endmodule
`default_nettype wire
