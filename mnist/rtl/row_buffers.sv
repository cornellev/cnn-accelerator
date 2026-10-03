// ================================================================
//
// Date  : October 3, 2026
// Author: Kaan Akan
//
// Memory file for the pixel row buffers. wr_data is written to the 
// first row's specified address while the rest of the values in that
// column address are shifted one row up with a cycle delay for BRAM mapping. 
// This means the provided wr_addr and wr_data should be delayed one column.
//
// Read is synchronous, rd_data gets the data in the
// column address a cycle later.
//
// ================================================================

module row_buffers #(
    parameter int unsigned PIXEL_WIDTH = 8,
    parameter int unsigned IMAGE_WIDTH = 320,
    parameter int unsigned ROWS = 2
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [COL_WIDTH-1:0]   wr_addr,
    input  logic [PIXEL_WIDTH-1:0] wr_data,
    input  logic                   wr_en,

    input  logic                   rd_en,
    input  logic [COL_WIDTH-1:0]   rd_addr,
    output logic [PIXEL_WIDTH-1:0] rd_data [0:ROWS-1]
);

    localparam int unsigned COL_WIDTH = (IMAGE_WIDTH > 1)  ? $clog2(IMAGE_WIDTH) : 1;
    logic [PIXEL_WIDTH-1:0] memory [0:ROWS-1] [0:IMAGE_WIDTH-1];

    // Read current column
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < ROWS; i++) begin
                rd_data[i] <= '0;
            end
        end
        else if (rd_en) begin
            for (int i = 0; i < ROWS; i++) begin
                rd_data[i] <= memory[i][rd_addr];
            end
        end
    end

    // Write new pixel and shift the old column values between rows
    // using data return in previous cycle's read
    always_ff @(posedge clk) begin
        if (wr_en && rst_n) begin
            memory[0][wr_addr] <= wr_data;
            for (int i = 0; i < ROWS-1; i++) begin
                memory[i+1][wr_addr] <= rd_data[i];
            end
        end
    end

endmodule
