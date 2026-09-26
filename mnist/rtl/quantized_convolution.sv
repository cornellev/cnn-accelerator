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

module quantized_convolution #(
    parameter int   M_BITS = 20,
    parameter logic        [5:0]        N     = 6'd10,
    parameter logic signed [7:0]        SHIFT = 8'sd1,
    parameter logic signed [M_BITS-1:0] M     = M_BITS'(10)
)(
    input  logic         [7:0] pixels  [0:8],
    input  logic signed  [7:0] weights [0:8],
    output logic signed  [7:0] quantized_value
);

    logic signed [19:0] unquantized_value;

    convolve_3x3 convolve
    (
        .pixels      (pixels),
        .weights     (weights),
        .convolution (unquantized_value)
    );

    quantized_scale #(
        .M_BITS (M_BITS),
        .M      (M),
        .N      (N),
        .SHIFT  (SHIFT)
    ) quantize
    (
        .convolution           (unquantized_value),
        .quantized_convolution (quantized_value)
    );

endmodule

`endif // _QUANTIZED_CONVOLUTION_SV_
