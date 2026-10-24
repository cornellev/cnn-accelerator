`include "rtl/quant_params.sv"

module convolve_3x3_fsm_minimal_tb;

    localparam int IMG_COLS   = 4;
    localparam int IMG_ROWS   = 4;
    localparam int NUM_PIXELS = IMG_ROWS * IMG_COLS;

    logic clk = 1'b0;
    always #5 clk = ~clk;

    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic pixel_valid = 1'b0;
    logic signed [ACTIVATION_BITS-1:0] input_pixel = '0;
    logic signed [WEIGHT_BITS-1:0] weights[0:0][0:8];
    logic signed [BIAS_BITS-1:0] bias[0:0];
    logic [SCALE_M_BITS-1:0] quant_scale_mult[0:0];
    logic signed [ACTIVATION_BITS-1:0] quant_offset = '0;
    logic signed [ACTIVATION_BITS-1:0] prev_quant_offset = '0;
    logic done, output_valid;
    logic signed [ACTIVATION_BITS-1:0] quantized_value[0:0];

    convolve_3x3_fsm #(
        .IMG_COLS(IMG_COLS),
        .IMG_ROWS(IMG_ROWS),
        .KERNEL_COUNT(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .pixel_valid(pixel_valid),
        .input_pixel(input_pixel),
        .weights(weights),
        .bias(bias),
        .quant_scale_mult(quant_scale_mult),
        .quant_offset(quant_offset),
        .prev_quant_offset(prev_quant_offset),
        .done(done),
        .output_valid(output_valid),
        .quantized_value(quantized_value)
    );

    // verilog_format: off
    logic signed [ACTIVATION_BITS-1:0] picture[0:NUM_PIXELS-1] = '{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16
    };

    // Same-size convolution with zero padding and a 3x3 all-ones kernel.
    logic signed [ACTIVATION_BITS-1:0] expected[0:NUM_PIXELS-1] = '{
        14, 24, 30, 22,
        33, 54, 63, 45,
        57, 90, 99, 69,
        46, 72, 78, 54
    };
    // verilog_format: on

    int output_count = 0;
    always @(posedge clk) begin
        if (rst_n && output_valid) begin
            if (output_count >= NUM_PIXELS) $fatal(1, "Unexpected extra output");
            if (quantized_value[0] !== expected[output_count])
                $fatal(1, "Output %0d: got %0d, expected %0d", output_count, quantized_value[0], expected[output_count]);
            output_count++;
        end
    end

    initial begin
        weights[0] = '{1, 1, 1, 1, 1, 1, 1, 1, 1};
        bias[0] = '0;
        quant_scale_mult[0] = SCALE_M_BITS'(1 << SCALE_SHIFT);  // Unity scale.

        repeat (3) @(posedge clk);
        rst_n <= 1'b1;

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        for (int i = 0; i < NUM_PIXELS; i++) begin
            pixel_valid <= 1'b1;
            input_pixel <= picture[i];
            @(posedge clk);
        end
        pixel_valid <= 1'b0;

        @(posedge done);
        @(posedge clk);
        #1;  // Let the output monitor finish before checking the count.
        if (output_count != NUM_PIXELS) $fatal(1, "Got %0d outputs, expected %0d", output_count, NUM_PIXELS);
        $display("CONVOLVE_3X3_FSM MINIMAL TB: PASS -- %0d outputs", output_count);
        $finish;
    end

    initial begin
        $dumpfile("convolve_3x3_fsm_minimal_tb.vcd");
        $dumpvars(0, convolve_3x3_fsm_minimal_tb);
    end

    initial begin
        #10000;
        $fatal(1, "Timeout waiting for frame completion");
    end

endmodule
