// ================================================================
//
// MNIST feature extractor: Conv+ReLU -> Pool -> Conv+ReLU -> Pool.
// Streams a 28x28x1 image into a 7x7x16 feature map.
//
// ================================================================

`ifndef _MNIST_SV_
`define _MNIST_SV_

`include "rtl/quant_params.sv"

module mnist (
    input logic clk,
    input logic rst_n,
    input logic start,

    input logic                              pixel_valid,
    input logic signed [ACTIVATION_BITS-1:0] input_pixel[0:LAYER_1_IN_CHANS-1],

    input logic signed [    WEIGHT_BITS-1:0] layer_1_weights          [0:LAYER_1_OUT_CHANS-1][0:LAYER_1_IN_CHANS-1][0:8],
    input logic signed [      BIAS_BITS-1:0] layer_1_bias             [0:LAYER_1_OUT_CHANS-1],
    input logic        [   SCALE_M_BITS-1:0] layer_1_quant_scale_mult [0:LAYER_1_OUT_CHANS-1],
    input logic signed [ACTIVATION_BITS-1:0] layer_1_quant_offset,
    input logic signed [ACTIVATION_BITS-1:0] layer_1_prev_quant_offset,

    input logic signed [    WEIGHT_BITS-1:0] layer_3_weights         [0:LAYER_3_OUT_CHANS-1][0:LAYER_3_IN_CHANS-1][0:8],
    input logic signed [      BIAS_BITS-1:0] layer_3_bias            [0:LAYER_3_OUT_CHANS-1],
    input logic        [   SCALE_M_BITS-1:0] layer_3_quant_scale_mult[0:LAYER_3_OUT_CHANS-1],
    input logic signed [ACTIVATION_BITS-1:0] layer_3_quant_offset,

    output logic                              done,
    output logic                              output_valid,
    output logic signed [ACTIVATION_BITS-1:0] output_pixel[0:LAYER_4_OUT_CHANS-1]
);

    localparam int LAYER_1_IN_COLS   = 28;
    localparam int LAYER_1_IN_ROWS   = 28;
    localparam int LAYER_1_IN_CHANS  = 1;
    localparam int LAYER_1_OUT_CHANS = 8;

    localparam int LAYER_2_IN_COLS   = LAYER_1_IN_COLS;
    localparam int LAYER_2_IN_ROWS   = LAYER_1_IN_ROWS;
    localparam int LAYER_2_IN_CHANS  = LAYER_1_OUT_CHANS;
    localparam int LAYER_2_OUT_CHANS = LAYER_2_IN_CHANS;

    localparam int LAYER_3_IN_COLS   = LAYER_2_IN_COLS / 2;
    localparam int LAYER_3_IN_ROWS   = LAYER_2_IN_ROWS / 2;
    localparam int LAYER_3_IN_CHANS  = LAYER_2_OUT_CHANS;
    localparam int LAYER_3_OUT_CHANS = 16;

    localparam int LAYER_4_IN_COLS   = LAYER_3_IN_COLS;
    localparam int LAYER_4_IN_ROWS   = LAYER_3_IN_ROWS;
    localparam int LAYER_4_IN_CHANS  = LAYER_3_OUT_CHANS;
    localparam int LAYER_4_OUT_CHANS = LAYER_4_IN_CHANS;

    // Layer 1: Conv2d(1, 8, 3, padding=1) + ReLU (fused in requantization).
    logic                              layer_1_pixel_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_1_input_pixel  [ 0:LAYER_1_IN_CHANS-1];
    logic                              layer_1_done;
    logic                              layer_1_output_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_1_output_pixel [0:LAYER_1_OUT_CHANS-1];

    assign layer_1_pixel_valid = pixel_valid;
    assign layer_1_input_pixel = input_pixel;

    convolve_3x3_3d_fsm #(
        .IMG_COLS     (LAYER_1_IN_COLS),
        .IMG_ROWS     (LAYER_1_IN_ROWS),
        .CHANNEL_COUNT(LAYER_1_IN_CHANS),
        .KERNEL_COUNT (LAYER_1_OUT_CHANS)
    ) layer_1_fsm (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .pixel_valid      (layer_1_pixel_valid),
        .input_pixel      (layer_1_input_pixel),
        .weights          (layer_1_weights),
        .bias             (layer_1_bias),
        .quant_scale_mult (layer_1_quant_scale_mult),
        .quant_offset     (layer_1_quant_offset),
        .prev_quant_offset(layer_1_prev_quant_offset),
        .done             (layer_1_done),
        .output_valid     (layer_1_output_valid),
        .quantized_value  (layer_1_output_pixel)
    );

    // Layer 2: MaxPool2d(2), 28x28x8 -> 14x14x8.
    logic                              layer_2_pixel_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_2_input_pixel  [ 0:LAYER_2_IN_CHANS-1];
    logic                              layer_2_done;
    logic                              layer_2_output_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_2_output_pixel [0:LAYER_2_OUT_CHANS-1];

    assign layer_2_pixel_valid = layer_1_output_valid;
    assign layer_2_input_pixel = layer_1_output_pixel;

    max_pooling_fsm #(
        .IMG_COLS     (LAYER_2_IN_COLS),
        .IMG_ROWS     (LAYER_2_IN_ROWS),
        .CHANNEL_COUNT(LAYER_2_IN_CHANS)
    ) layer_2_fsm (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .pixel_valid (layer_2_pixel_valid),
        .input_pixel (layer_2_input_pixel),
        .done        (layer_2_done),
        .output_valid(layer_2_output_valid),
        .max_value   (layer_2_output_pixel)
    );

    // Layer 3: Conv2d(8, 16, 3, padding=1) + ReLU (fused in requantization).
    logic                              layer_3_pixel_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_3_input_pixel       [ 0:LAYER_3_IN_CHANS-1];
    logic signed [ACTIVATION_BITS-1:0] layer_3_prev_quant_offset;
    logic                              layer_3_done;
    logic                              layer_3_output_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_3_output_pixel      [0:LAYER_3_OUT_CHANS-1];

    assign layer_3_pixel_valid       = layer_2_output_valid;
    assign layer_3_input_pixel       = layer_2_output_pixel;
    // Max pooling preserves the preceding convolution's quantization.
    assign layer_3_prev_quant_offset = layer_1_quant_offset;

    convolve_3x3_3d_fsm #(
        .IMG_COLS     (LAYER_3_IN_COLS),
        .IMG_ROWS     (LAYER_3_IN_ROWS),
        .CHANNEL_COUNT(LAYER_3_IN_CHANS),
        .KERNEL_COUNT (LAYER_3_OUT_CHANS)
    ) layer_3_fsm (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .pixel_valid      (layer_3_pixel_valid),
        .input_pixel      (layer_3_input_pixel),
        .weights          (layer_3_weights),
        .bias             (layer_3_bias),
        .quant_scale_mult (layer_3_quant_scale_mult),
        .quant_offset     (layer_3_quant_offset),
        .prev_quant_offset(layer_3_prev_quant_offset),
        .done             (layer_3_done),
        .output_valid     (layer_3_output_valid),
        .quantized_value  (layer_3_output_pixel)
    );

    // Layer 4: MaxPool2d(2), 14x14x16 -> 7x7x16.
    logic                              layer_4_pixel_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_4_input_pixel  [ 0:LAYER_4_IN_CHANS-1];
    logic                              layer_4_done;
    logic                              layer_4_output_valid;
    logic signed [ACTIVATION_BITS-1:0] layer_4_output_pixel [0:LAYER_4_OUT_CHANS-1];

    assign layer_4_pixel_valid = layer_3_output_valid;
    assign layer_4_input_pixel = layer_3_output_pixel;

    max_pooling_fsm #(
        .IMG_COLS     (LAYER_4_IN_COLS),
        .IMG_ROWS     (LAYER_4_IN_ROWS),
        .CHANNEL_COUNT(LAYER_4_IN_CHANS)
    ) layer_4_fsm (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .pixel_valid (layer_4_pixel_valid),
        .input_pixel (layer_4_input_pixel),
        .done        (layer_4_done),
        .output_valid(layer_4_output_valid),
        .max_value   (layer_4_output_pixel)
    );

    assign done = layer_4_done;
    assign output_valid = layer_4_output_valid;
    assign output_pixel = layer_4_output_pixel;

endmodule

`endif  // _MNIST_SV_
