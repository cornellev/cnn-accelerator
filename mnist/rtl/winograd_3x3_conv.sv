// ================================================================
//
// Date  : Oct 6, 2026
// Author: Ishaan Parikh
//
// Combinational block for winograd convolution
//
// Uses the winograd convolution unit to go through a 3x3 kernel of pixel values.
//
// ================================================================

module winograd_3x3_conv
(
    input  logic         [7:0] tile        [0:15],  // 4x4 pixels, row-major
    input  logic signed [11:0] u_weights   [0:15],  // G' g G'^T, precomputed
    output logic signed [19:0] convolution [0:3]    // 2x2 outputs, row-major
);

    logic signed [10:0] d  [0:15];  // zero-extended pixels
    logic signed [10:0] bt [0:15];  // B^T d
    logic signed [10:0] v  [0:15];  // B^T d B
    logic signed [22:0] m  [0:15];  // U' (.) V
    logic signed [24:0] at [0:7];   // A^T m      (2x4)
    logic signed [24:0] y  [0:3];   // A^T m A    (2x2) = 4 x convolution

    always_comb begin
        for (int i = 0; i < 16; i++) begin
            d[i] = $signed({3'b0, tile[i]});
        end

        // Input transform, rows: B^T d
        for (int c = 0; c < 4; c++) begin
            bt[0*4+c] = d[0*4+c] - d[2*4+c];
            bt[1*4+c] = d[1*4+c] + d[2*4+c];
            bt[2*4+c] = d[2*4+c] - d[1*4+c];
            bt[3*4+c] = d[1*4+c] - d[3*4+c];
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
            convolution[i] = 20'(y[i] >>> 2);
        end
    end

endmodule