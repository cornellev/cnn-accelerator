// ================================================================
//
// Date  : Oct 31, 2026
// Author: Kaan Akan
//
// FSM for the max pooling of N channels.
//
// ================================================================

`include "rtl/quant_params.sv"

module max_pooling_fsm #(
    parameter IMG_COLS      = 320,
    parameter IMG_ROWS      = 320,
    parameter CHANNEL_COUNT = 8

) (
    input logic clk,
    input logic rst_n,
    input logic start,

    input logic                               pixel_valid,
    input logic signed [ACTIVATION_BITS-1:0]  input_pixel [0:CHANNEL_COUNT-1],

    output logic                              output_valid,
    output logic                              done,
    output logic signed [ACTIVATION_BITS-1:0] max_value [0:CHANNEL_COUNT-1]
);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_BUSY
    } state_t;

    state_t current_state, next_state;

    // State register
    always_ff @(posedge clk) begin : state_register
        if (!rst_n) begin
            current_state <= ST_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    assign done = accept_pixel && (row_count == IMG_ROWS - 1) && (col_count == IMG_COLS - 1);

    // Next state logic
    always_comb begin : next_state_logic
        next_state = current_state;
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

    logic busy;
    logic row_col_count_enable;

    // Output logic
    always_comb begin : output_logic
        case (current_state)
            ST_IDLE: begin
                busy                 = 1'b0;
                row_col_count_enable = 1'b0;
            end

            ST_BUSY: begin
                busy                 = 1'b1;
                // only increment counter for valid pixels
                row_col_count_enable = pixel_valid;
            end

            default: begin
                busy                 = 1'b0;
                row_col_count_enable = 1'b0;
            end
        endcase
    end

    localparam ROW_COUNT_WIDTH = $clog2(IMG_ROWS);
    localparam COL_COUNT_WIDTH = $clog2(IMG_COLS);

    logic [ROW_COUNT_WIDTH-1:0] row_count;
    logic [COL_COUNT_WIDTH-1:0] col_count;

    always_ff @(posedge clk) begin : row_col_counter
        if (!rst_n || start) begin
            row_count <= '0;
            col_count <= '0;
        end else if (row_col_count_enable) begin
            if ((row_count == IMG_ROWS - 1) && (col_count == IMG_COLS - 1)) begin
                row_count <= '0;
                col_count <= '0;
            end else if (col_count == IMG_COLS - 1) begin
                row_count <= row_count + 1'b1;
                col_count <= '0;
            end else begin
                row_count <= row_count;
                col_count <= col_count + 1'b1;
            end
        end
    end

    logic        [COL_COUNT_WIDTH-1:0] col_count_delayed;
    logic signed [ACTIVATION_BITS-1:0] input_pixel_delayed[0:CHANNEL_COUNT-1];
    logic                              accept_pixel_delayed;
    logic                              accept_pixel;

    assign accept_pixel = busy && pixel_valid;

    always_ff @(posedge clk) begin : buffer_delay_register
        if (!rst_n) begin
            col_count_delayed    <= '0;
            accept_pixel_delayed <= '0;
            for (int c = 0; c < CHANNEL_COUNT; c++) begin
                input_pixel_delayed[c] <= '0;
            end
        end else begin
            accept_pixel_delayed <= accept_pixel;
            if (accept_pixel) begin
                col_count_delayed   <= col_count;
                input_pixel_delayed <= input_pixel;
            end
        end

    end

    logic signed [ACTIVATION_BITS-1:0] pixel_row_read[0:CHANNEL_COUNT-1][0];

    // One set of row buffers per channel, all using the same addresses
    for (genvar c = 0; c < CHANNEL_COUNT; c++) begin : generate_channel_row_buffers
        row_buffers #(
            .IMAGE_WIDTH(IMG_COLS),
            .ROWS       (1)
        ) pixel_row_buffers (
            .clk    (clk),
            .rst_n  (rst_n),
            .wr_addr(col_count_delayed),
            .wr_data(input_pixel_delayed[c]),
            .wr_en  (accept_pixel_delayed),
            .rd_en  (accept_pixel),
            .rd_addr(col_count),
            .rd_data(pixel_row_read[c])
        );
    end

    // Window valid check
    logic window_ready_delayed;

    always_ff @(posedge clk) begin : window_center_valid
        if (!rst_n || start) begin
            window_ready_delayed <= 1'b0;
            output_valid         <= 1'b0;
        end else begin
            window_ready_delayed <= accept_pixel && (row_count >= 1) && (col_count >= 1);
            output_valid <= window_ready_delayed;
        end
    end
    logic signed [ACTIVATION_BITS-1:0] window_pixels[0:CHANNEL_COUNT-1][0:3];

    always_ff @(posedge clk) begin : channel_window_register
        if (!rst_n || start) begin
            for (int c = 0; c < CHANNEL_COUNT; c++) begin
                for (int i = 0; i < 4; i++) begin
                    window_pixels[c][i] <= '0;
                end
            end
        end else if (accept_pixel_delayed) begin
            for (int c = 0; c < CHANNEL_COUNT; c++) begin
                window_pixels[c][0] <= window_pixels[c][1];
                window_pixels[c][1] <= pixel_row_read[c][0];
                window_pixels[c][2] <= window_pixels[c][3];
                window_pixels[c][3] <= input_pixel_delayed[c];

            end
        end
    end

    for (genvar i = 0; i < CHANNEL_COUNT; i++) begin : generate_max_pools
        max_pooling pool 
        (
            .input_pixel_1 (window_pixels[i][0]),
            .input_pixel_2 (window_pixels[i][1]),
            .input_pixel_3 (window_pixels[i][2]),
            .input_pixel_4 (window_pixels[i][3]),
            .max_pixel     (max_value[i])
        );
    end

endmodule