// ================================================================
//
// Date  : Sep 26, 2026
// Author: Kaan Akan
//
// Quantization rescaling for the MNIST project.
//
// Takes in the 20-bit accumulated convolution value and calculates
// the quantized convolution with the following:
// SHIFT + ((M * convolution) / (2 ^ N))
//
// ================================================================

`ifndef _QUANT_SCALE_SV_
`define _QUANT_SCALE_SV_

module quant_scale #(
    parameter int   M_BITS = 20,
    parameter logic        [5:0]        N     = 6'd10,
    parameter logic signed [7:0]        SHIFT = 8'sd1,
    parameter logic signed [M_BITS-1:0] M     = M_BITS'(10)
)(
    input  logic signed [19:0] convolution,
    output logic signed [7:0]  quantized_convolution
);

    logic signed [M_BITS+19:0] scaled_value;

    always_comb begin
        scaled_value = SHIFT + ((M * convolution) >>> N);

        if (scaled_value > 127) begin
            quantized_convolution =  8'sd127;
        end
        else if (scaled_value < -128) begin
            quantized_convolution = -8'sd128;
        end
        else begin
            quantized_convolution = scaled_value[7:0];
        end
    end

endmodule

`endif // _QUANT_SCALE_SV_
