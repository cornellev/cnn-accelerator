// ================================================================
//
// Date  : Sep 29, 2026
// Author: Kaan Akan
//
// Quantizatized convolution unit's coordinator for the MNIST project.
//
// Uses the convolution unit to go through a 2D array of pixel values.
//
// ================================================================

module convolve_3x3_fsm #(
    parameter IMG_COLS = 320,
    parameter IMG_ROWS = 320
)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic [7:0]         pixels,
    output logic signed  [7:0] quantized_value
);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_BUSY
    } state_t;

    state_t current_state, next_state;

    // State register
    always_ff @(posedge clk) begin : state_register
        if (!rst_n) begin
            current_state <= '0;
        end
        else begin
            current_state <= next_state;
        end
    end

    logic  done;
    assign done = (row_count == IMG_ROWS - 1) && (col_count == IMG_COLS - 1);

    // Next state logic
    always_comb begin : next_state_logic
        case (current_state)
            ST_IDLE: begin
                if (start) begin
                    next_state = ST_BUSY;
                end
            end

            ST_BUSY: begin
                if (done) begin
                    next_state = ST_IDLE;
                end
            end

        default: next_state = ST_IDLE;
        endcase
    end

    logic row_col_count_enable;

    // Output logic
    always_comb begin : output_logic
        case (current_state) 
            ST_IDLE: begin
                row_col_count_enable = 1'b0;
            end

            ST_BUSY: begin
                row_col_count_enable = 1'b1;
            end

            default: enable = 1'b0;
        endcase
    end

    localparam ROW_COUNT_WIDTH = $clog2(IMAGE_ROWS);
    localparam COL_COUNT_WIDTH = $clog2(IMAGE_COLS);

    logic [ROW_COUNT_WIDTH-1:0] row_count;
    logic [COL_COUNT_WIDTH-1:0] col_count;

    always_ff @(posedge clk) begin : row_col_counter
        if (!rst_n || start) begin
            row_count <= '0;
            col_count <= '0;
        end
        else if (row_col_count_enable) begin
            if ((row_count == IMG_ROWS - 1) && (col_count == IMG_COLS - 1)) begin
                row_count <= '0;
                col_count <= '0;
            end
            else if (col_count == IMG_COLS - 1) begin
                row_count <= row_count + 1'b1;
                col_count <= '0;
            end
        end
    end

endmodule

