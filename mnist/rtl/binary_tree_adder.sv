// ================================================================
//
// Date  : Nov 7, 2026
// Author: Kaan Akan
//
// Parameterizable binary addition tree.
// Pads the number of elements up to the closest power of two.
//
// ================================================================

`ifndef _BINARY_TREE_ADDER_SV_
`define _BINARY_TREE_ADDER_SV_

`include "rtl/quant_params.sv"

module binary_tree_adder #(
    parameter int N = 49,
    parameter int IN_BITS = ACTIVATION_BITS + WEIGHT_BITS
) (
    input  logic signed [IN_BITS-1:0] addends [0:N-1],
    output logic signed [IN_BITS+$clog2(N)-1:0] sum
);

    localparam int CEIL_POWER_TWO_EXP = $clog2(N);
    localparam int CEIL_POWER_TWO     = 2 ** CEIL_POWER_TWO_EXP;
    localparam int SUM_BITS           = IN_BITS + CEIL_POWER_TWO_EXP;

    logic signed [SUM_BITS-1:0] node [0:(2*CEIL_POWER_TWO)-2];

    always_comb begin
        // Initialize upper array nodes with addend values and zeros
        // to pad to closest power of two   
        for (int i = 0; i < CEIL_POWER_TWO; i++) begin
            if (i < N) begin
                node[i+(CEIL_POWER_TWO-1)] = SUM_BITS'(addends[i]);
            end
            else begin
                node[i+(CEIL_POWER_TWO-1)] = '0;
            end
        end

        // Complete addition
        for (int j = CEIL_POWER_TWO-2; j >= 0; j--) begin
            node[j] = node[2*j+1] + node[2*j+2];
        end
    end

    assign sum = SUM_BITS'(node[0]);

endmodule

`endif  // _BINARY_TREE_ADDER_SV_
