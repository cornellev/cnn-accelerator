// ================================================================
//
// Date  : Sep 26, 2026
// Author: Kaan Akan
//
// 3x3 convolution unit with quantized output for the MNIST project.
//
// ================================================================

`ifndef _QUANTIZED_CONVOLUTION_SV_
`define _QUANTIZED_CONVOLUTION_SV_

`include "rtl/quant_params.sv"

module quantized_convolution #(
    parameter logic RELU_ON = 1
) (
    input  logic signed [ACTIVATION_BITS-1:0] pixels          [0:8],
    input  logic signed [    WEIGHT_BITS-1:0] weights         [0:8],
    input  logic signed [      BIAS_BITS-1:0] full_bias,
    input  logic        [   SCALE_M_BITS-1:0] quant_scale_mult,
    input  logic signed [ACTIVATION_BITS-1:0] quant_offset,
    output logic signed [ACTIVATION_BITS-1:0] quantized_value
);

    localparam int CONVOLUTION_BITS = ACTIVATION_BITS + WEIGHT_BITS + $clog2(9);
    // One carry bit for adding the signed convolution and bias (27 by default).
    localparam int ACCUMULATOR_BITS = ((BIAS_BITS > CONVOLUTION_BITS) ? BIAS_BITS : CONVOLUTION_BITS) + 1;

    logic signed [CONVOLUTION_BITS-1:0] unquantized_value;
    logic signed [ACCUMULATOR_BITS-1:0] biased_value;

    assign biased_value = ACCUMULATOR_BITS'(unquantized_value) + ACCUMULATOR_BITS'(full_bias);

    convolve_3x3 convolve (
        .pixels     (pixels),
        .weights    (weights),
        .convolution(unquantized_value)
    );

    quantized_scale #(
        .INPUT_BITS(ACCUMULATOR_BITS),
        .RELU_ON(RELU_ON)
    ) quantize (
        .convolution          (biased_value),
        .quant_scale_mult     (quant_scale_mult),
        .quant_offset         (quant_offset),
        .quantized_convolution(quantized_value)
    );

endmodule

`endif  // _QUANTIZED_CONVOLUTION_SV_
