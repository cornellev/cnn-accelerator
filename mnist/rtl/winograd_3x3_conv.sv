// ================================================================
//
// Date  : Oct 6, 2026
// Author: Ishaan Parikh
//
// Combinational block for winograd convolution
//
// Computes the 2x2 block of 3x3 convolutions in a 4x4 tile of signed
// pixels with 16 multiplies, using U' from winograd_weight_transform.
// Output matches four convolve_3x3 calls exactly.
//
// ================================================================

`ifndef _WINOGRAD_3X3_CONV_SV_
`define _WINOGRAD_3X3_CONV_SV_

`include "rtl/quant_params.sv"

module winograd_3x3_conv (
    input  logic signed [                      ACTIVATION_BITS-1:0] tile       [0:15],  // 4x4 pixels, row-major
    input  logic signed [                        WEIGHT_BITS+4-1:0] u_weights  [0:15],  // G' g G'^T
    output logic signed [ACTIVATION_BITS+WEIGHT_BITS+$clog2(9)-1:0] convolution[ 0:3]   // 2x2 outputs, row-major
);

    localparam int BT_BITS = ACTIVATION_BITS + 1;  // |B^T d|   <= 2 * 128
    localparam int V_BITS = ACTIVATION_BITS + 2;  // |V|       <= 4 * 128
    localparam int M_BITS = WEIGHT_BITS + 4 + V_BITS;  // |U' V|  <= 1152 * 512
    localparam int AT_BITS = M_BITS + 2;  // sum of 3 products
    localparam int Y_BITS = AT_BITS + 2;  // sum of 3 partial sums
    localparam int CONVOLUTION_BITS = ACTIVATION_BITS + WEIGHT_BITS + $clog2(9);

    logic signed [BT_BITS-1:0] bt[0:15];  // B^T d
    logic signed [ V_BITS-1:0] v [0:15];  // B^T d B
    logic signed [ M_BITS-1:0] m [0:15];  // U' (.) V
    logic signed [AT_BITS-1:0] at[ 0:7];  // A^T m      (2x4)
    logic signed [ Y_BITS-1:0] y [ 0:3];  // A^T m A    (2x2) = 4 x convolution

    always_comb begin
        // Input transform, rows: B^T d
        for (int c = 0; c < 4; c++) begin
            bt[0*4+c] = tile[0*4+c] - tile[2*4+c];
            bt[1*4+c] = tile[1*4+c] + tile[2*4+c];
            bt[2*4+c] = tile[2*4+c] - tile[1*4+c];
            bt[3*4+c] = tile[1*4+c] - tile[3*4+c];
        end

        // Input transform, columns: (B^T d) B
        for (int r = 0; r < 4; r++) begin
            v[r*4+0] = bt[r*4+0] - bt[r*4+2];
            v[r*4+1] = bt[r*4+1] + bt[r*4+2];
            v[r*4+2] = bt[r*4+2] - bt[r*4+1];
            v[r*4+3] = bt[r*4+1] - bt[r*4+3];
        end

        // The only 16 multiplies
        for (int i = 0; i < 16; i++) begin
            m[i] = u_weights[i] * v[i];
        end

        // Output transform, rows: A^T m
        for (int c = 0; c < 4; c++) begin
            at[0*4+c] = m[0*4+c] + m[1*4+c] + m[2*4+c];
            at[1*4+c] = m[1*4+c] - m[2*4+c] - m[3*4+c];
        end

        // Output transform, columns: (A^T m) A
        for (int r = 0; r < 2; r++) begin
            y[r*2+0] = at[r*4+0] + at[r*4+1] + at[r*4+2];
            y[r*2+1] = at[r*4+1] - at[r*4+2] - at[r*4+3];
        end

        // U' used 2G, so y is exactly 4x the direct result
        for (int i = 0; i < 4; i++) begin
            convolution[i] = CONVOLUTION_BITS'(y[i] >>> 2);
        end
    end

endmodule

`endif  // _WINOGRAD_3X3_CONV_SV_
