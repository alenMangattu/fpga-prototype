// Small circuit that connects read/write requests to sync_ram.
// Hold inputs stable before the rising clock edge.
module ram_access #(
    parameter int ADDR_WIDTH = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic                  clk,
    input  logic                  write_en,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [DATA_WIDTH-1:0] write_data,
    input  logic                  read_en,
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [DATA_WIDTH-1:0] read_data
);
    logic                  ram_cs;
    logic                  ram_we;
    logic [ADDR_WIDTH-1:0] ram_addr;

    // One-port RAM: a write takes priority if both requests are high.
    assign ram_cs   = write_en || read_en;
    assign ram_we   = write_en;
    assign ram_addr = write_en ? write_addr : read_addr;

    sync_ram #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) ram (
        .clk(clk),
        .cs(ram_cs),
        .we(ram_we),
        .addr(ram_addr),
        .write_data(write_data),
        .read_data(read_data)
    );
endmodule
