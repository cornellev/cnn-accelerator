// ================================================================
//
// Date  : Oct 31, 2026
// Author: Kaan Akan
//
// Returns maximum value of 4 input pixels
//
// ================================================================

`ifndef _MAX_POOLING_SV_
`define _MAX_POOLING_SV_

module max_pooling
(
    input  logic signed [ACTIVATION_BITS-1:0] input_pixel_1,
    input  logic signed [ACTIVATION_BITS-1:0] input_pixel_2,
    input  logic signed [ACTIVATION_BITS-1:0] input_pixel_3,
    input  logic signed [ACTIVATION_BITS-1:0] input_pixel_4,

    output logic signed [ACTIVATION_BITS-1:0] max_pixel
);

    logic [ACTIVATION_BIT-1:0] inter_1, inter_2;

    assign inter_1   = (input_pixel_1 > input_pixel_2) ? 
                        input_pixel_1 : input_pixel_2;
    assign inter_2   = (input_pixel_3 > input_pixel_4) ? 
                        input_pixel_3 : input_pixel_4;

    assign max_pixel = (inter_1 > inter_2) ? inter_1 : inter_2;

endmodule

`endif  // _MAX_POOLING_SV_