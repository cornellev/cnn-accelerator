// ================================================================
//
// Date  : Sep 26, 2026
// Author: Kaan Akan
//
// Quantization rescaling for the MNIST project.
//
// Requantizes a bias-added convolution using a runtime multiplier and
// output zero point.
// Matches quantize_custom.py: ReLU, multiply, fixed shift with rounding
// (ties toward positive infinity), add quant_offset, then saturate.
//
// ================================================================

`ifndef _QUANTIZED_SCALE_SV_
`define _QUANTIZED_SCALE_SV_

`include "rtl/quant_params.sv"

module quantized_scale #(
    parameter int INPUT_BITS,
    parameter logic RELU_ON = 1
) (
    input  logic signed [     INPUT_BITS-1:0] convolution,
    input  logic        [   SCALE_M_BITS-1:0] quant_scale_mult,
    input  logic signed [ACTIVATION_BITS-1:0] quant_offset,
    output logic signed [ACTIVATION_BITS-1:0] quantized_convolution
);

    localparam int PRODUCT_BITS = INPUT_BITS + SCALE_M_BITS + 1;
    localparam logic signed [ACTIVATION_BITS-1:0] QUANT_MAX = {1'b0, {(ACTIVATION_BITS - 1) {1'b1}}};
    localparam logic signed [ACTIVATION_BITS-1:0] QUANT_MIN = {1'b1, {(ACTIVATION_BITS - 1) {1'b0}}};

    logic signed [  INPUT_BITS-1:0] relu_value;
    logic signed [PRODUCT_BITS-1:0] product;
    logic signed [PRODUCT_BITS-1:0] rounded_value, scaled_value;

    assign relu_value = (RELU_ON && (convolution < 0)) ? '0 : convolution;

    // Zero-extend quant_scale_mult before using the signed multiplier.
    assign product = $signed({1'b0, quant_scale_mult}) * relu_value;

    if (SCALE_SHIFT > 0) begin : round_scaled
        logic rounding_bit;

        // The highest discarded bit determines whether to round up.
        assign rounding_bit  = product[SCALE_SHIFT-1];
        assign rounded_value = PRODUCT_BITS'(product >>> SCALE_SHIFT) + PRODUCT_BITS'(rounding_bit);
    end else begin : no_shift
        assign rounded_value = product;
    end

    assign scaled_value = rounded_value + PRODUCT_BITS'(quant_offset);

    always_comb begin
        if (scaled_value > PRODUCT_BITS'(QUANT_MAX)) begin
            quantized_convolution = QUANT_MAX;
        end else if (scaled_value < PRODUCT_BITS'(QUANT_MIN)) begin
            quantized_convolution = QUANT_MIN;
        end else begin
            quantized_convolution = scaled_value[ACTIVATION_BITS-1:0];
        end
    end

endmodule

`endif  // _QUANTIZED_SCALE_SV_
