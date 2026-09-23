// Educational DRAM-like memory model for simulation.
// One outstanding request; fixed response latency; no refresh or DDR timing.
module fake_dram #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32,
    parameter int LATENCY = 4   // clock edges from accepted request to response
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  req_valid,
    output logic                  req_ready,
    input  logic                  req_write,
    input  logic [ADDR_WIDTH-1:0] req_addr,
    input  logic [DATA_WIDTH-1:0] req_wdata,
    output logic                  resp_valid,
    output logic [DATA_WIDTH-1:0] resp_rdata
);
    localparam int COUNT_WIDTH = (LATENCY > 1) ? $clog2(LATENCY) : 1;

    logic [DATA_WIDTH-1:0] mem [0:(1 << ADDR_WIDTH)-1];
    logic busy;
    logic [COUNT_WIDTH-1:0] count;
    logic saved_write;
    logic [ADDR_WIDTH-1:0] saved_addr;
    logic [DATA_WIDTH-1:0] saved_wdata;

    assign req_ready = !busy;

    initial begin
        if (LATENCY < 1) $fatal(1, "LATENCY must be at least 1");
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            count      <= '0;
            resp_valid <= 1'b0;
            resp_rdata <= '0;
            // Like real DRAM after power-up, mem contents are not initialized.
        end else begin
            resp_valid <= 1'b0; // one-clock completion pulse

            if (!busy) begin
                if (req_valid) begin
                    saved_write <= req_write;
                    saved_addr  <= req_addr;
                    saved_wdata <= req_wdata;
                    count       <= COUNT_WIDTH'(LATENCY - 1);
                    busy        <= 1'b1;
                end
            end else if (count != 0) begin
                count <= count - 1'b1;
            end else begin
                if (saved_write)
                    mem[saved_addr] <= saved_wdata;
                else
                    resp_rdata <= mem[saved_addr];
                resp_valid <= 1'b1;
                busy       <= 1'b0;
            end
        end
    end
endmodule
