// ================================================================
//
// Date  : Sep 12, 2026
// Author: Kaan Akan
//
// 3 x 3 convolution module for the MNIST project.
//
// ================================================================

`ifndef _CONVOLVE_3X3_SV_
`define _CONVOLVE_3X3_SV_

`include "rtl/quant_params.sv"

module convolve_3x3 (
    input  logic signed [                      ACTIVATION_BITS-1:0] pixels     [0:8],
    input  logic signed [                          WEIGHT_BITS-1:0] weights    [0:8],
    output logic signed [ACTIVATION_BITS+WEIGHT_BITS+$clog2(9)-1:0] convolution
);

    localparam int PRODUCT_BITS = ACTIVATION_BITS + WEIGHT_BITS;
    localparam int CONVOLUTION_BITS = PRODUCT_BITS + $clog2(9);

    logic signed [PRODUCT_BITS-1:0] intermediate_product[0:8];

    always_comb begin
        convolution = '0;

        for (int i = 0; i < 9; i++) begin
            intermediate_product[i] = pixels[i] * weights[i];
            convolution = convolution + CONVOLUTION_BITS'(intermediate_product[i]);
        end
    end

endmodule

`endif  // _CONVOLVE_3X3_SV_
