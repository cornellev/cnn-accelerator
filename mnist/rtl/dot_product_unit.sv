// ================================================================
//
// Date  : Nov 7, 2026
// Author: Kaan Akan
//
// N-element dot product unit for the MNIST project.
// Defaults to N = 49, one 7x7 channel of the feature map going into
// the classifier.
//
// ================================================================

`ifndef _DOT_PRODUCT_UNIT_SV_
`define _DOT_PRODUCT_UNIT_SV_

`include "rtl/quant_params.sv"

module dot_product_unit #(
    parameter N = 49,
    parameter ROW_WIDTH = 784
) (
    input  logic signed [                      ACTIVATION_BITS-1:0] activations [0:N-1],
    input  logic signed [                          WEIGHT_BITS-1:0] weights     [0:N-1],
    output logic signed [ACTIVATION_BITS+WEIGHT_BITS+$clog2(ROW_WIDTH)-1:0] dot_product
);

    logic signed [ACTIVATION_BITS+WEIGHT_BITS-1:0] multiplies [0:N-1];

    always_comb begin
        // Element-wise multiply
        for (int i = 0; i < N; i++) begin
            multiplies[i] = activations[i] * weights[i];
        end

        // Sum element-wise multiplications
        dot_product = '0;
        for (int j = 0; j < N; j++) begin
            dot_product = dot_product + $bits(dot_product)'(multiplies[j]);
        end
    end

endmodule

`endif  // _DOT_PRODUCT_UNIT_SV_
