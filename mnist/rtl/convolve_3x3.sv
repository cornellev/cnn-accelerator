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

module convolve_3x3 
(
    input  logic         [7:0] pixels  [0:8],
    input  logic signed  [7:0] weights [0:8],
    output logic signed [19:0] convolution
);

    logic signed [15:0] intermediate_product [0:8];

    always_comb begin
        convolution = '0;

        for (int i = 0; i < 9; i++) begin
            intermediate_product[i] = $signed({1'b0, pixels[i]}) * weights[i];
            convolution = convolution + 20'(intermediate_product[i]);
        end
    end

endmodule

`endif // _CONVOLVE_3X3_SV_
