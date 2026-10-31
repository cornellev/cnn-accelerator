// ================================================================
//
// Date  : Oct 3, 2026
// Author: Kaan Akan
//
// Quantizatized convolution unit's coordinator for the MNIST project.
//
// Uses the convolution unit to go through a 2D array of pixel values.
//
// ================================================================

`include "rtl/quant_params.sv"

module convolve_3x3_fsm #(
    parameter IMG_COLS     = 320,
    parameter IMG_ROWS     = 320,
    parameter KERNEL_COUNT = 32
) (
    input logic clk,
    input logic rst_n,
    input logic start,

    input logic                              pixel_valid,
    input logic signed [ACTIVATION_BITS-1:0] input_pixel,

    input logic signed [WEIGHT_BITS-1:0] weights[0:KERNEL_COUNT-1][0:8],
    input logic signed [  BIAS_BITS-1:0] bias   [0:KERNEL_COUNT-1],

    input logic        [   SCALE_M_BITS-1:0] quant_scale_mult [0:KERNEL_COUNT-1],
    input logic signed [ACTIVATION_BITS-1:0] quant_offset,
    input logic signed [ACTIVATION_BITS-1:0] prev_quant_offset,

    output logic                              output_valid,
    output logic                              done,
    output logic signed [ACTIVATION_BITS-1:0] quantized_value[0:KERNEL_COUNT-1]
);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_BUSY_PREFILL,
        ST_BUSY_COMPUTE,
        ST_BUSY_FINISH
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

    always_comb begin : next_state_logic
        next_state = current_state;
        case (current_state)
            ST_IDLE: begin
                if (start) begin
                    next_state = ST_BUSY_PREFILL;
                end
            end

            ST_BUSY_PREFILL: begin
                if (next_prefill_done) begin
                    next_state = ST_BUSY_COMPUTE;
                end
            end

            ST_BUSY_COMPUTE: begin
                if (is_last_input) begin
                    next_state = ST_BUSY_FINISH;
                end
            end

            ST_BUSY_FINISH: begin
                if (done) begin
                    next_state = ST_IDLE;
                end
            end

            default: next_state = ST_IDLE;
        endcase
    end

    logic advance_buffer;
    always_comb begin : output_logic
        case (current_state)
            ST_IDLE: begin
                advance_buffer = '0;
            end

            ST_BUSY_PREFILL: begin
                advance_buffer = pixel_valid;
            end

            ST_BUSY_COMPUTE: begin
                advance_buffer = pixel_valid;
            end

            ST_BUSY_FINISH: begin
                advance_buffer = '1;
            end

            default: begin
                advance_buffer = '0;
            end
        endcase
    end

    // We need some extra bits here because buffer_row
    // will proceed past the end of the image
    localparam ROW_COUNT_WIDTH = $clog2(IMG_ROWS + 2);
    localparam COL_COUNT_WIDTH = $clog2(IMG_COLS);

    localparam logic [ROW_COUNT_WIDTH-1:0] LAST_ROW = ROW_COUNT_WIDTH'(IMG_ROWS - 1);
    localparam logic [COL_COUNT_WIDTH-1:0] LAST_COL = COL_COUNT_WIDTH'(IMG_COLS - 1);

    logic [ROW_COUNT_WIDTH-1:0] buffer_row;
    logic [COL_COUNT_WIDTH-1:0] buffer_col;
    logic [ROW_COUNT_WIDTH-1:0] buffer_row_delayed;
    logic [COL_COUNT_WIDTH-1:0] buffer_col_delayed;

    always_ff @(posedge clk) begin : buffer_pos
        if (!rst_n || start) begin
            buffer_row <= '0;
            buffer_col <= '0;
            buffer_row_delayed <= '0;
            buffer_col_delayed <= '0;
        end else if (advance_buffer) begin
            if (buffer_col == LAST_COL) begin
                buffer_row <= buffer_row + 1'b1;
                buffer_col <= '0;
            end else begin
                buffer_row <= buffer_row;
                buffer_col <= buffer_col + 1'b1;
            end

            buffer_row_delayed <= buffer_row;
            buffer_col_delayed <= buffer_col;
        end
    end

    // When buffer_col becomes 1 and buffer_row becomes 1, two cycles later we
    // can start moving the window. So next_prefill_done is based on
    // buffer_row and buffer_col delayed by one cycle. This drives state
    // transitions in the FSM.
    logic next_prefill_done;
    assign next_prefill_done = advance_buffer_delayed && buffer_row_delayed == 1 && buffer_col_delayed == 1;

    logic is_last_input;
    assign is_last_input = advance_buffer && buffer_col == LAST_COL && buffer_row == LAST_ROW;

    assign done = window_valid && window_row == LAST_ROW && window_col == LAST_COL;

    logic advance_buffer_delayed;
    logic signed [ACTIVATION_BITS-1:0] input_pixel_delayed;
    always_ff @(posedge clk) begin : buffer_delay_register
        if (!rst_n || start) begin
            input_pixel_delayed <= '0;
            advance_buffer_delayed <= '0;
        end else begin
            advance_buffer_delayed <= advance_buffer;
            if (advance_buffer) begin
                input_pixel_delayed <= input_pixel;
            end
        end
    end

    logic signed [ACTIVATION_BITS-1:0] pixel_row_read[0:1];

    row_buffers #(
        .IMAGE_WIDTH(IMG_COLS),
        .ROWS       (2)
    ) pixel_row_buffers (
        .clk    (clk),
        .rst_n  (rst_n),
        .wr_addr(buffer_col_delayed),
        .wr_data(input_pixel_delayed),
        .wr_en  (advance_buffer_delayed),
        .rd_en  (advance_buffer),
        .rd_addr(buffer_col),
        .rd_data(pixel_row_read)
    );

    logic signed [ACTIVATION_BITS-1:0] window_pixels[0:8];
    always_ff @(posedge clk) begin : kernel_1_window_register
        if (!rst_n || start) begin
            for (int i = 0; i < 9; i++) begin
                window_pixels[i] <= '0;
            end
        end else if (advance_buffer_delayed) begin
            window_pixels[0] <= window_pixels[1];
            window_pixels[1] <= window_pixels[2];
            window_pixels[2] <= pixel_row_read[1];

            window_pixels[3] <= window_pixels[4];
            window_pixels[4] <= window_pixels[5];
            window_pixels[5] <= pixel_row_read[0];

            window_pixels[6] <= window_pixels[7];
            window_pixels[7] <= window_pixels[8];
            window_pixels[8] <= input_pixel_delayed;
        end
    end

    logic next_window_valid;
    assign next_window_valid = advance_buffer_delayed && (next_state == ST_BUSY_COMPUTE || next_state == ST_BUSY_FINISH);

    logic window_valid;
    always_ff @(posedge clk) begin : window_valid_update
        if (!rst_n || start) begin
            window_valid <= '0;
        end else begin
            window_valid <= next_window_valid;
        end
    end

    assign output_valid = window_valid;

    // These are technically oversized but it doesn't matter much
    logic [ROW_COUNT_WIDTH-1:0] window_row;
    logic [COL_COUNT_WIDTH-1:0] window_col;
    always_ff @(posedge clk) begin : window_pos
        if (!rst_n || start) begin
            window_row <= '0;
            window_col <= '0;
        end else if (next_window_valid) begin
            if (buffer_col_delayed == 0) begin
                // Special case: we sometimes need to wrap around to the
                // previous row
                window_row <= buffer_row_delayed - 2;
                window_col <= LAST_COL;
            end else begin
                // Normally, window lags two cycles behind buffer and is
                // 1 unit up and to the left
                window_row <= buffer_row_delayed - 1;
                window_col <= buffer_col_delayed - 1;
            end
        end
    end

    // We need to mask invalid pixels by assigning zero point of previous layer
    logic signed [ACTIVATION_BITS-1:0] masked_window_pixels[0:8];

    always_comb begin : mask_window
        for (int i = 0; i < 9; i++) begin
            masked_window_pixels[i] = window_pixels[i];
        end

        if (window_row == 0) begin
            masked_window_pixels[0] = prev_quant_offset;
            masked_window_pixels[1] = prev_quant_offset;
            masked_window_pixels[2] = prev_quant_offset;
        end

        if (window_col == 0) begin
            masked_window_pixels[0] = prev_quant_offset;
            masked_window_pixels[3] = prev_quant_offset;
            masked_window_pixels[6] = prev_quant_offset;
        end

        if (window_row == LAST_ROW) begin
            masked_window_pixels[6] = prev_quant_offset;
            masked_window_pixels[7] = prev_quant_offset;
            masked_window_pixels[8] = prev_quant_offset;
        end

        if (window_col == LAST_COL) begin
            masked_window_pixels[2] = prev_quant_offset;
            masked_window_pixels[5] = prev_quant_offset;
            masked_window_pixels[8] = prev_quant_offset;
        end
    end

    for (genvar i = 0; i < KERNEL_COUNT; i++) begin : generate_kernel
        quantized_convolution kernel (
            .pixels          (masked_window_pixels),
            .weights         (weights[i]),
            .full_bias       (bias[i]),
            .quant_scale_mult(quant_scale_mult[i]),
            .quant_offset    (quant_offset),
            .quantized_value (quantized_value[i])
        );
    end

endmodule
