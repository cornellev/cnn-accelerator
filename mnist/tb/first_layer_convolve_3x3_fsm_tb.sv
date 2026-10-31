// ================================================================
//
// Date  : Oct 17, 2026
// Author: Kaan Akan
//
// Testbench for the 3x3 quantized convolution FSM for 8 kernel
// and directly inputted weights, which matches the first layer MNIST.
//
// Streams a picture into the FSM, pushes every output to a scoreboard
// when output_valid is high, then checks the scoreboard against the
// expected outputs once the frame is done.
//
// ================================================================

`ifndef _CONVOLVE_3X3_FSM_TB_SV_
`define _CONVOLVE_3X3_FSM_TB_SV_

`include "rtl/quant_params.sv"

module first_layer_convolve_3x3_fsm_tb;

    localparam int IMG_COLS     = 28;
    localparam int IMG_ROWS     = 28;
    localparam int KERNEL_COUNT = 8;

    localparam int NUM_PIXELS = IMG_ROWS * IMG_COLS;

    localparam string PICTURE_PATH = "tb/vectors/first_layer_input.txt";
    localparam string RESULT_PATH  = "tb/vectors/first_layer_output.txt";

    logic                        clk;
    logic                        rst_n;
    logic                        start;
    logic                        pixel_valid;
    logic        [          7:0] input_pixel;
    logic signed [          7:0] weights     [0:KERNEL_COUNT-1] [0:8];
    logic signed [BIAS_BITS-1:0] bias        [0:KERNEL_COUNT-1];

    logic        [   SCALE_M_BITS-1:0] quant_scale_mult  [0:KERNEL_COUNT-1];
    logic signed [ACTIVATION_BITS-1:0] quant_offset;
    logic signed [ACTIVATION_BITS-1:0] prev_quant_offset;

    logic              done;
    logic              output_valid;
    logic signed [7:0] quantized_value[0:KERNEL_COUNT-1];

    convolve_3x3_fsm #(
        .IMG_COLS    (IMG_COLS),
        .IMG_ROWS    (IMG_ROWS),
        .KERNEL_COUNT(KERNEL_COUNT)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .pixel_valid      (pixel_valid),
        .input_pixel      (input_pixel),
        .weights          (weights),
        .bias             (bias),
        .quant_scale_mult (quant_scale_mult),
        .quant_offset     (quant_offset),
        .prev_quant_offset(prev_quant_offset),
        .done             (done),
        .output_valid     (output_valid),
        .quantized_value  (quantized_value)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    logic        [                               7:0] picture [NUM_PIXELS];
    logic signed [(ACTIVATION_BITS*KERNEL_COUNT)-1:0] expected[NUM_PIXELS];

    // Every valid output gets pushed here, checked after the frame is done
    logic signed [(ACTIVATION_BITS*KERNEL_COUNT)-1:0] scoreboard[$];

    logic [(ACTIVATION_BITS*KERNEL_COUNT)-1:0] packed_outputs;

    always @(posedge clk) begin
        if (output_valid) begin
            // Kernel 0 in the top byte, kernel 7 in the bottom byte
            packed_outputs = {>>{quantized_value}};
            scoreboard.push_back(packed_outputs);
        end
    end

    // Sends the picture one pixel per cycle with pixel_valid high
    task send_frame();
        for (int i = 0; i < NUM_PIXELS; i++) begin
            @(negedge clk);
            pixel_valid = 1'b1;
            input_pixel = picture[i];
        end
        @(negedge clk);
        pixel_valid = 1'b0;
    endtask

    integer num_errors;

    initial begin
        $readmemb(PICTURE_PATH, picture);
        $readmemb(RESULT_PATH, expected);

        // First layer's integer inference buffers from the MNIST model,
        // matching model/export_image.py and the first_layer vectors.
        // Don't have a good pipeline for this yet other than looking at
        // the exported .mem file or inspecting the model through Python.
        weights[0] = '{-8'sd22, -8'sd63, -8'sd128, -8'sd55, -8'sd47, 8'sd35, 8'sd70, 8'sd111, 8'sd63};
        weights[1] = '{8'sd54, -8'sd63, -8'sd79, -8'sd128, -8'sd119, -8'sd13, -8'sd63, 8'sd36, 8'sd116};
        weights[2] = '{-8'sd61, -8'sd17, 8'sd73, 8'sd107, 8'sd34, 8'sd119, 8'sd24, 8'sd59, 8'sd127};
        weights[3] = '{-8'sd22, -8'sd17, -8'sd85, -8'sd54, 8'sd9, -8'sd127, -8'sd120, -8'sd93, -8'sd74};
        weights[4] = '{-8'sd19, 8'sd127, 8'sd93, 8'sd110, 8'sd60, -8'sd28, 8'sd65, -8'sd105, -8'sd120};
        weights[5] = '{-8'sd39, 8'sd87, 8'sd101, 8'sd0, 8'sd52, 8'sd106, 8'sd127, 8'sd85, 8'sd51};
        weights[6] = '{8'sd127, 8'sd70, 8'sd123, -8'sd82, -8'sd42, 8'sd10, -8'sd58, -8'sd54, -8'sd120};
        weights[7] = '{8'sd69, 8'sd96, 8'sd43, 8'sd64, 8'sd82, 8'sd96, 8'sd127, -8'sd10, -8'sd11};

        // full_bias includes the input-zero-point correction; do not use raw biases.
        bias = '{2563, -15626, 77748, -41946, 27742, 69488, 1310, 71199};
        quant_scale_mult = '{151, 124, 97, 88, 95, 106, 103, 113};
        quant_offset = ACTIVATION_BITS'(-128);
        prev_quant_offset = ACTIVATION_BITS'(-128);

        rst_n       = 1'b0;
        start       = 1'b0;
        pixel_valid = 1'b0;
        input_pixel = '0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        send_frame();

        @(posedge done);
        @(posedge clk);
        @(negedge clk);

        num_errors = 0;
        if (scoreboard.size() != NUM_PIXELS) begin
            $display("MISS got %0d outputs, want %0d", scoreboard.size(), NUM_PIXELS);
            num_errors = num_errors + 1;
        end
        for (int i = 0; i < NUM_PIXELS && i < scoreboard.size(); i++) begin
            if (scoreboard[i] !== expected[i]) begin
                num_errors = num_errors + 1;
                if (num_errors <= 20)
                    $display(
                        "MISS output %0d (row %0d, col %0d) | got %0d | want %0d",
                        i,
                        i / IMG_COLS,
                        i % IMG_COLS,
                        scoreboard[i],
                        expected[i]
                    );
            end
        end

        if (num_errors == 0) begin
            $display("CONVOLVE_3x3_FSM TB: PASS -- %0d outputs, 0 errors", NUM_PIXELS);
            $finish;
        end else begin
            $fatal(1, "CONVOLVE_3x3_FSM TB: FAIL -- %0d outputs, %0d errors", NUM_PIXELS, num_errors);
        end
    end

    initial begin
        repeat (2 * NUM_PIXELS + 4 * IMG_COLS + 100) @(posedge clk);
        $fatal(1, "CONVOLVE_3x3_FSM TB: timed out waiting for frame completion");
    end

endmodule

`endif  // _CONVOLVE_3X3_FSM_TB_SV_
