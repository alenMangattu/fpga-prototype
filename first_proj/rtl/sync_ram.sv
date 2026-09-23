// Simple single-port synchronous RAM model.
// DEPTH = 2**ADDR_WIDTH words; DATA_WIDTH bits per word.
// Read data changes on a rising clock edge and holds when cs is low.
// If cs and we are both high, write only; read_data holds its old value.
// Contents and read_data are unknown until written/read (as with real SRAM).
module sync_ram #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic                  clk,
    input  logic                  cs,         // chip select
    input  logic                  we,         // write enable
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] write_data,
    output logic [DATA_WIDTH-1:0] read_data
);
    logic [DATA_WIDTH-1:0] mem [0:(1 << ADDR_WIDTH)-1];

    always_ff @(posedge clk) begin
        if (cs) begin
            if (we)
                mem[addr] <= write_data;
            else
                read_data <= mem[addr];
        end
    end
endmodule
