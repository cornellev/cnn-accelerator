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

module convolve_3x3_fsm #(
    parameter IMG_COLS = 320,
    parameter IMG_ROWS = 320,
    parameter PIXEL_WIDTH = 8

)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               start,
    input  logic               pixel_valid,
    input  logic         [7:0] input_pixel,

    output logic               output_valid,
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
            current_state <= ST_IDLE;
        end
        else begin
            current_state <= next_state;
        end
    end

    logic  done;
    assign done = accept_pixel && (row_count == IMG_ROWS - 1) && 
                                  (col_count == IMG_COLS - 1);

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
            else begin
                row_count <= row_count;
                col_count <= col_count + 1'b1;
            end
        end
    end

    logic [COL_COUNT_WIDTH-1:0] col_count_delayed;
    logic [7:0]                 input_pixel_delayed;

    logic  accept_pixel_delayed;
    logic  accept_pixel;
    assign accept_pixel = busy && pixel_valid;

    always_ff @(posedge clk) begin : buffer_delay_register
        if (!rst_n) begin
            col_count_delayed    <= '0;
            input_pixel_delayed  <= '0;
            accept_pixel_delayed <= '0;
        end
        else begin
            accept_pixel_delayed <= accept_pixel;
            if (accept_pixel) begin
            col_count_delayed    <= col_count;
            input_pixel_delayed  <= input_pixel;
            end
        end

    end
    
    logic [7:0] pixel_row_read [0:1];

    row_buffers #(
        .PIXEL_WIDTH (PIXEL_WIDTH),
        .IMAGE_WIDTH (IMG_COLS),
        .ROWS        (2)
    ) pixel_row_buffers
    (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_addr (col_count_delayed),
        .wr_data (input_pixel_delayed),
        .wr_en   (accept_pixel_delayed),
        .rd_en   (accept_pixel),
        .rd_addr (col_count),
        .rd_data (pixel_row_read)
    );

    // Hardcoded weight for kernel 1 for now
    logic signed  [7:0] weights_1 [0:8];

    always_comb begin : harcode_kernel_1_weights
        weights_1[0] = -8'd60;
        weights_1[1] = -8'd45;
        weights_1[2] = -8'd30;
        weights_1[3] = -8'd15;
        weights_1[4] = -8'd0;
        weights_1[6] =  8'd15;
        weights_1[5] =  8'd30;
        weights_1[7] =  8'd45;
        weights_1[8] =  8'd60;
    end

    // Window valid check
    logic window_ready_delayed;

    always_ff @(posedge clk) begin : window_center_valid
        if (!rst_n || start) begin
            window_ready_delayed <= 1'b0;
            output_valid         <= 1'b0;
        end else begin
            window_ready_delayed <= accept_pixel && (row_count >= 2) &&
                                                    (col_count >= 2);
            output_valid <= window_ready_delayed;
        end
    end
    logic [7:0] window_pixels [0:8];

    always_ff @(posedge clk) begin : kernel_1_window_register
        if (!rst_n || start) begin
            for (int i = 0; i < 9; i++) begin
                window_pixels[i] <= '0;
            end
        end
        else if (accept_pixel_delayed) begin
            window_pixels[0] <= window_pixels[1];
            window_pixels[1] <= window_pixels[2];
            window_pixels[2] <= pixel_row_read[0];

            window_pixels[3] <= window_pixels[4];
            window_pixels[4] <= window_pixels[5];
            window_pixels[5] <= pixel_row_read[1];
            
            window_pixels[6] <= window_pixels[7];
            window_pixels[7] <= window_pixels[8];
            window_pixels[8] <= input_pixel_delayed;
        end
    end

    quantized_convolution kernel_1
    (
        .pixels          (window_pixels),
        .weights         (weights_1),
        .quantized_value (quantized_value)
    );
    
endmodule

