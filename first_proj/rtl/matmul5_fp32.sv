// Sequential 5x5 FP32 matrix multiplier, row-major order.
// Each output is computed with five rounded FP32 multiplies and adds.
// Arithmetic modules: https://github.com/dawsonjon/fpu (MIT license).
module matmul5_fp32 (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        wr_valid,
    output logic        wr_ready,
    input  logic        wr_bank,   // 0 = A, 1 = B
    input  logic [4:0]  wr_addr,   // row * 5 + column, 0..24
    input  logic [31:0] wr_data,   // IEEE-754 binary32 bit pattern

    input  logic        cmd_valid,
    output logic        cmd_ready,

    output logic        result_valid,
    input  logic        result_ready,
    output logic [4:0]  result_addr,
    output logic [31:0] result_data
);
    typedef enum logic [2:0] {
        IDLE, MUL_A, MUL_B, MUL_WAIT, ADD_A, ADD_B, ADD_WAIT, EMIT
    } state_t;

    state_t state;
    logic [31:0] a_mem [0:24];
    logic [31:0] b_mem [0:24];
    logic [24:0] a_written, b_written;
    logic [2:0] row, col, k;
    logic [31:0] acc, product;

    logic mul_a_ack, mul_b_ack, mul_z_stb;
    logic [31:0] mul_z;
    logic add_a_ack, add_b_ack, add_z_stb;
    logic [31:0] add_z;

    wire [4:0] a_index = row * 5 + k;
    wire [4:0] b_index = k * 5 + col;

    assign wr_ready = (state == IDLE) && (wr_addr < 25);
    assign cmd_ready = (state == IDLE) && (&a_written) && (&b_written)
                       && !wr_valid;
    assign result_valid = (state == EMIT);
    assign result_addr = row * 5 + col;
    assign result_data = acc;

    multiplier fp_mul (
        .clk(clk), .rst(!rst_n),
        .input_a(a_mem[a_index]), .input_a_stb(state == MUL_A),
        .input_a_ack(mul_a_ack),
        .input_b(b_mem[b_index]), .input_b_stb(state == MUL_B),
        .input_b_ack(mul_b_ack),
        .output_z(mul_z), .output_z_stb(mul_z_stb),
        .output_z_ack(state == MUL_WAIT)
    );

    adder fp_add (
        .clk(clk), .rst(!rst_n),
        .input_a(acc), .input_a_stb(state == ADD_A),
        .input_a_ack(add_a_ack),
        .input_b(product), .input_b_stb(state == ADD_B),
        .input_b_ack(add_b_ack),
        .output_z(add_z), .output_z_stb(add_z_stb),
        .output_z_ack(state == ADD_WAIT)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            a_written <= '0;
            b_written <= '0;
            row <= '0;
            col <= '0;
            k <= '0;
            acc <= '0;
            product <= '0;
        end else begin
            case (state)
                IDLE: begin
                    if (wr_valid && wr_ready) begin
                        if (wr_bank) begin
                            b_mem[wr_addr] <= wr_data;
                            b_written[wr_addr] <= 1'b1;
                        end else begin
                            a_mem[wr_addr] <= wr_data;
                            a_written[wr_addr] <= 1'b1;
                        end
                    end else if (cmd_valid && cmd_ready) begin
                        row <= 0;
                        col <= 0;
                        k <= 0;
                        acc <= 32'h00000000; // +0.0
                        state <= MUL_A;
                    end
                end
                MUL_A: if (mul_a_ack) state <= MUL_B;
                MUL_B: if (mul_back) state <= MUL_WAIT;
                MUL_WAIT: if (mul_z_stb) begin
                    product <= mul_z;
                    state <= ADD_A;
                end
                ADD_A: if (add_a_ack) state <= ADD_B;
                ADD_B: if (add_b_ack) state <= ADD_WAIT;
                ADD_WAIT: if (add_z_stb) begin
                    acc <= add_z;
                    if (k == 4) state <= EMIT;
                    else begin
                        k <= k + 1'b1;
                        state <= MUL_A;
                    end
                end
                EMIT: if (result_ready) begin
                    if (row == 4 && col == 4) state <= IDLE;
                    else begin
                        if (col == 4) begin
                            col <= 0;
                            row <= row + 1'b1;
                        end else col <= col + 1'b1;
                        k <= 0;
                        acc <= 32'h00000000;
                        state <= MUL_A;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
