// ================================================================
//
// Date  : Oct 17, 2026
// Author: Kaan Akan
//
// Testbench for the 3x3 quantized convolution FSM for one kernel
// and directly inputted weights.
//
// Streams a picture into the FSM, pushes every output to a scoreboard
// when output_valid is high, then checks the scoreboard against the
// expected outputs once the frame is done.
//
// ================================================================

`ifndef _CONVOLVE_3X3_FSM_TB_SV_
`define _CONVOLVE_3X3_FSM_TB_SV_

module convolve_3x3_fsm_tb;

    localparam int IMG_COLS     = 314;
    localparam int IMG_ROWS     = 314;
    localparam int PIXEL_WIDTH  = 8;
    localparam int KERNEL_COUNT = 1;

    localparam int NUM_PIXELS  = IMG_ROWS * IMG_COLS;
    localparam int NUM_OUTPUTS = (IMG_ROWS - 2) * (IMG_COLS - 2);

    localparam string PICTURE_PATH = "tb/picture/calibration_board.txt";
    localparam string RESULT_PATH  = "tb/vectors/quantized_convolution_calibration_board_vectors.txt";

    logic               clk;
    logic               rst_n;
    logic               start;
    logic               pixel_valid;
    logic         [7:0] input_pixel;
    logic signed  [7:0] weights [0:KERNEL_COUNT-1] [0:8];

    logic               done;
    logic               output_valid;
    logic signed  [7:0] quantized_value [0:KERNEL_COUNT-1];

    convolve_3x3_fsm #(
        .IMG_COLS     (IMG_COLS),
        .IMG_ROWS     (IMG_ROWS),
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .KERNEL_COUNT (KERNEL_COUNT)
    ) dut
    (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (start),
        .pixel_valid     (pixel_valid),
        .input_pixel     (input_pixel),
        .weights         (weights),
        .done            (done),
        .output_valid    (output_valid),
        .quantized_value (quantized_value)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    logic        [7:0] picture  [0:NUM_PIXELS-1];
    logic signed [7:0] expected [0:NUM_OUTPUTS-1];

    // Every valid output gets pushed here, checked after the frame is done
    logic signed [7:0] scoreboard [$];

    always @(posedge clk) begin
        if (output_valid) begin
            scoreboard.push_back(quantized_value[0]);
        end
    end

    // Sends the picture one pixel per cycle with pixel_valid high
    task send_frame();
        for (int i = 0; i < NUM_PIXELS; i++) begin
            pixel_valid <= 1'b1;
            input_pixel <= picture[i];
            @(posedge clk);
        end
        pixel_valid <= 1'b0;
    endtask

    integer num_errors;

    initial begin
        $readmemb(PICTURE_PATH, picture);
        $readmemb(RESULT_PATH,  expected);

        weights[0] = '{-8'sd60, -8'sd45, -8'sd30,
                       -8'sd15,  8'sd0,  8'sd30,
                        8'sd15,  8'sd45, 8'sd60};

        rst_n       = 1'b0;
        start       = 1'b0;
        pixel_valid = 1'b0;
        input_pixel = '0;

        repeat (3) @(posedge clk);
        rst_n <= 1'b1;

        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        send_frame();

        // The last output comes out a couple of cycles after done
        repeat (5) @(posedge clk);

        num_errors = 0;
        if (scoreboard.size() != NUM_OUTPUTS) begin
            $display("MISS got %0d outputs, want %0d", scoreboard.size(), NUM_OUTPUTS);
            num_errors = num_errors + 1;
        end
        for (int i = 0; i < NUM_OUTPUTS && i < scoreboard.size(); i++) begin
            if (scoreboard[i] !== expected[i]) begin
                num_errors = num_errors + 1;
                if (num_errors <= 20)
                    $display("MISS output %0d (row %0d, col %0d) | got %0d | want %0d",
                             i, i / (IMG_COLS - 2), i % (IMG_COLS - 2), scoreboard[i], expected[i]);
            end
        end

        if (num_errors == 0) begin
            $display("CONVOLVE_3x3_FSM TB: PASS -- %0d outputs, 0 errors", NUM_OUTPUTS);
            $finish;
        end
        else begin
            $fatal(1, "CONVOLVE_3x3_FSM TB: FAIL -- %0d outputs, %0d errors",
                   NUM_OUTPUTS, num_errors);
        end
    end

endmodule

`endif // _CONVOLVE_3X3_FSM_TB_SV_
