// ================================================================
//
// Date  : Oct 24, 2026
// Author: Kaan Akan
//
// K-stacked 3x3 convolution units to do 3D convolution (3x3xK).
//
// ================================================================

`ifndef _QUANTIZED_CONVOLUTION_3D_SV_
`define _QUANTIZED_CONVOLUTION_3D_SV_

`include "rtl/quant_params.sv"

module quantized_convolution_3d #(
    parameter logic RELU_ON = 1,
    parameter int   K       = 32
) (
    input  logic signed [ACTIVATION_BITS-1:0] pixels          [0:K-1][0:8],
    input  logic signed [    WEIGHT_BITS-1:0] weights         [0:K-1][0:8],
    input  logic signed [      BIAS_BITS-1:0] full_bias,
    input  logic        [   SCALE_M_BITS-1:0] quant_scale_mult,
    input  logic signed [ACTIVATION_BITS-1:0] quant_offset,
    output logic signed [ACTIVATION_BITS-1:0] quantized_value
);

    localparam int CONVOLUTION_BITS = ACTIVATION_BITS + WEIGHT_BITS + $clog2(9);
    // One carry bit for adding the signed convolution and bias (27 by default).
    localparam int ACCUMULATOR_BITS = ((BIAS_BITS > CONVOLUTION_BITS + $clog2(K)) ? BIAS_BITS : CONVOLUTION_BITS + $clog2(K)) + 1;

    logic signed [ACCUMULATOR_BITS-1:0] biased_value;

    logic signed [CONVOLUTION_BITS-1:0] unquantized_value_arr [0:K-1];

    // Do the 1D convolutions
    for (genvar k = 0; k < K; k++) begin : generate_kernels
        convolve_3x3 convolve (
            .pixels     (pixels[k]),
            .weights    (weights[k]),
            .convolution(unquantized_value_arr[k])
        );
    end

    localparam int EXPANSION_ADDITION_BITS = $clog2(K);
    localparam int CONVOLUTION_SUM_BITS    = CONVOLUTION_BITS + EXPANSION_ADDITION_BITS;
    logic signed [CONVOLUTION_SUM_BITS-1:0] unquantized_value_sum;

    // Add the 1D convolution results together
    always_comb begin
        unquantized_value_sum = '0;
        for (int k = 0; k < K; k++) begin
            unquantized_value_sum = unquantized_value_sum + CONVOLUTION_SUM_BITS'(unquantized_value_arr[k]);
        end
    end

    assign biased_value = ACCUMULATOR_BITS'(unquantized_value_sum) + ACCUMULATOR_BITS'(full_bias);

    quantized_scale #(
        .INPUT_BITS (ACCUMULATOR_BITS),
        .RELU_ON    (RELU_ON)
    ) quantize (
        .convolution          (biased_value),
        .quant_scale_mult     (quant_scale_mult),
        .quant_offset         (quant_offset),
        .quantized_convolution(quantized_value)
    );

endmodule

`endif  // _QUANTIZED_CONVOLUTION_3D_SV_
