// ================================================================
//
// Date  : Oct 17, 2026
// Author: Ishaan Parikh
//
// Combinational Winograd F(2x2, 3x3) kernel transform.
//
// Turns a raw 3x3 kernel g into the 4x4 U' = G' g G'^T that
// winograd_3x3_conv takes, so only the raw weights need storing.
// G' = 2G keeps U' an integer, so it's only adds and constant shifts.
// Runs once per kernel load, so one instance can be shared.
//
// ================================================================

`ifndef _WINOGRAD_WEIGHT_TRANSFORM_SV_
`define _WINOGRAD_WEIGHT_TRANSFORM_SV_

`include "rtl/quant_params.sv"

module winograd_weight_transform (
    input  logic signed [  WEIGHT_BITS-1:0] weights  [ 0:8],  // g, 3x3 row-major
    output logic signed [WEIGHT_BITS+4-1:0] u_weights[0:15]   // U' = G' g G'^T, 4x4 row-major
);

    localparam int T_BITS = WEIGHT_BITS + 2;  // |G' g|        <= 3 * 128
    localparam int U_BITS = WEIGHT_BITS + 4;  // |G' g G'^T|   <= 9 * 128

    logic signed [T_BITS-1:0] gc[ 0:2];  // g[0][c] + g[2][c], shared by rows 1 and 2
    logic signed [T_BITS-1:0] t [0:11];  // G' g        (4x3)
    logic signed [U_BITS-1:0] tr[ 0:3];  // t[r][0] + t[r][2], shared by cols 1 and 2

    always_comb begin
        // Rows: G' g
        for (int c = 0; c < 3; c++) begin
            gc[c]    = weights[0*3+c] + weights[2*3+c];
            t[0*3+c] = weights[0*3+c] <<< 1;
            t[1*3+c] = gc[c] + weights[1*3+c];
            t[2*3+c] = gc[c] - weights[1*3+c];
            t[3*3+c] = weights[2*3+c] <<< 1;
        end

        // Columns: (G' g) G'^T
        for (int r = 0; r < 4; r++) begin
            tr[r]            = t[r*3+0] + t[r*3+2];
            u_weights[r*4+0] = t[r*3+0] <<< 1;
            u_weights[r*4+1] = tr[r] + t[r*3+1];
            u_weights[r*4+2] = tr[r] - t[r*3+1];
            u_weights[r*4+3] = t[r*3+2] <<< 1;
        end
    end

endmodule

`endif  // _WINOGRAD_WEIGHT_TRANSFORM_SV_
