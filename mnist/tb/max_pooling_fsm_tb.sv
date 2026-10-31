// ================================================================
//
// Date  : Oct 31, 2026
// Author: Kaan Akan
//
// Testbench for the 2x2 max pooling FSM. Uses one channel to
// match the single-channel MNIST picture and expected vectors.
//
// Streams a picture into the FSM, pushes every pooled output to a
// scoreboard when output_valid is high, then checks the scoreboard
// against the expected outputs once the frame is done.
//
// ================================================================

`ifndef _MAX_POOLING_FSM_TB_SV_
`define _MAX_POOLING_FSM_TB_SV_

`include "rtl/quant_params.sv"

module max_pooling_fsm_tb;

    localparam int IMG_COLS = 28;
    localparam int IMG_ROWS = 28;
    localparam int CHANNEL_COUNT = 1;

    localparam int  INPUT_NUM_PIXELS = IMG_ROWS * IMG_COLS;
    localparam int OUTPUT_NUM_PIXELS = (IMG_ROWS / 2) * (IMG_COLS / 2);

    localparam string PICTURE_PATH = "tb/picture/mnist-4.txt";
    localparam string RESULT_PATH  = "tb/vectors/mnist-4-maxpool-output.txt";

    logic                              clk;
    logic                              rst_n;
    logic                              start;
    logic                              pixel_valid;
    logic signed [ACTIVATION_BITS-1:0] input_pixel       [0:CHANNEL_COUNT-1];

    logic                              done;
    logic                              output_valid;
    logic signed [ACTIVATION_BITS-1:0] max_value [0:CHANNEL_COUNT-1];

    max_pooling_fsm # (
        .IMG_COLS      (IMG_COLS),
        .IMG_ROWS      (IMG_ROWS),
        .CHANNEL_COUNT (CHANNEL_COUNT)
    ) dut
    (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .pixel_valid  (pixel_valid),
        .input_pixel  (input_pixel),
        .done         (done),
        .output_valid (output_valid),
        .max_value    (max_value)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    logic signed [ACTIVATION_BITS-1:0] picture [0:INPUT_NUM_PIXELS-1];
    logic signed [ACTIVATION_BITS-1:0] expected[0:OUTPUT_NUM_PIXELS-1];

    // Every valid output gets pushed here, checked after the frame is done
    logic signed [ACTIVATION_BITS-1:0] scoreboard[$];

    always @(posedge clk) begin
        if (output_valid) begin
            scoreboard.push_back(max_value[0]);
        end
    end

    // Sends the picture one pixel per cycle with pixel_valid high
    task send_frame();
        for (int i = 0; i < INPUT_NUM_PIXELS; i++) begin
            @(negedge clk);
            pixel_valid = 1'b1;
            input_pixel[0] = picture[i];
        end
        @(negedge clk);
        pixel_valid = 1'b0;
    endtask

    integer num_errors;

    initial begin
        $readmemb(PICTURE_PATH, picture);
        $readmemb(RESULT_PATH, expected);

        rst_n       = 1'b0;
        start       = 1'b0;
        pixel_valid = 1'b0;
        input_pixel = '{default: '0};

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        send_frame();

        // done accompanies the last valid output. Let the posedge monitor
        // capture it, then inspect the queue away from that sampling edge.
        @(posedge done);
        @(posedge clk);
        @(negedge clk);

        for (int i = 0; i < 100; i++) begin
            @(posedge clk);
        end

        num_errors = 0;
        if (scoreboard.size() != OUTPUT_NUM_PIXELS) begin
            $display("MISS got %0d outputs, want %0d", scoreboard.size(), OUTPUT_NUM_PIXELS);
            num_errors = num_errors + 1;
        end
        for (int i = 0; i < OUTPUT_NUM_PIXELS && i < scoreboard.size(); i++) begin
            if (scoreboard[i] !== expected[i]) begin
                num_errors = num_errors + 1;
                if (num_errors <= 20)
                    $display(
                        "MISS output %0d (row %0d, col %0d) | got %0d | want %0d",
                        i,
                        i / (IMG_COLS / 2),
                        i % (IMG_COLS / 2),
                        scoreboard[i],
                        expected[i]
                    );
            end
        end

        if (num_errors == 0) begin
            $display("MAX_POOLING_FSM TB: PASS -- %0d outputs, 0 errors", OUTPUT_NUM_PIXELS);
            $finish;
        end else begin
            $fatal(1, "MAX_POOLING_FSM TB: FAIL -- %0d outputs, %0d errors", OUTPUT_NUM_PIXELS, num_errors);
        end
    end

    initial begin
        repeat (2 * INPUT_NUM_PIXELS + 4 * IMG_COLS + 100) @(posedge clk);
        $fatal(1, "MAX_POOLING_FSM TB: timed out waiting for frame completion");
    end

endmodule

`endif  // _MAX_POOLING_FSM_TB_SV_
