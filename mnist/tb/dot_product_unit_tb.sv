// ================================================================
//
// Date  : Nov 15, 2026
// Author: Ishaan Parikh
//
// Testbench for the dot product  unit using a reference model.
// using reference model which has directed and random  vectors.
//
//
// ================================================================

`ifndef _DOT_PRODUCT_UNIT_TB_SV_
`define _DOT_PRODUCT_UNIT_TB_SV_

`include "rtl/quant_params.sv"

module dot_product_unit_tb;

    localparam int N = 49;
    localparam int ROW_WIDTH = 784;
    localparam int OUT_BITS  = ACTIVATION_BITS + WEIGHT_BITS + $clog2(ROW_WIDTH);
    localparam int NUM_RANDOM = 1000;

    logic signed [ACTIVATION_BITS-1:0] activations [0:N-1];
    logic signed [    WEIGHT_BITS-1:0] weights     [0:N-1];
    logic signed [       OUT_BITS-1:0] dot_product;

    dot_product_unit #(
        .N          (N),
        .ROW_WIDTH  (ROW_WIDTH)
    ) dut (
        .activations (activations),
        .weights     (weights),
        .dot_product (dot_product)
    );

    integer num_tests, num_errors;

    //Reference model
    function automatic int expected_dot_product();
        int sum;
        sum = 0;
        for (int i = 0; i < N; i++) begin
            sum = sum + activations[i] * weights[i];
        end
        return sum;
    endfunction

    task automatic check(input string name);
        int expected;
        #1;
        expected = expected_dot_product();
        num_tests = num_tests + 1;
        if (dot_product !== expected) begin
            num_errors = num_errors + 1;
            //Displaying first 20 errors
            if (num_errors <= 20)
                $display("MISS %s | got %0d | want %0d", name, dot_product, expected);
        end
    endtask

        //helper function for simplifying filling in inputs.
        task automatic fill(input int act_value, input int weight_value);
        for (int i = 0; i < N; i++) begin
            activations[i] = ACTIVATION_BITS'(act_value);
            weights[i]     = WEIGHT_BITS'(weight_value);
        end
    endtask
    initial begin
        num_tests  = 0;
        num_errors = 0;
        // directed
        fill(0, 0);       check("all zeros");          // 0
        fill(1, 1);       check("all ones");           // 49
        fill(-1, 1);      check("negative ones");      // -49
        fill(-128, -128); check("max positive");       // 49 * 16384 = 802816
        fill(-128, 127);  check("max negative");       // 49 * -16256 = -796544

        // One element at a time
        for (int p = 0; p < N; p++) begin
            fill(0, 0);
            activations[p] = 1;
            weights[p]     = WEIGHT_BITS'(p + 1);
            check($sformatf("only element %0d", p));  // p + 1
        end

        // Random vectors
        for (int t = 0; t < NUM_RANDOM; t++) begin
            for (int i = 0; i < N; i++) begin
                activations[i] = $urandom();
                weights[i]     = $urandom();
            end
            check($sformatf("random %0d", t));
        end

        if (num_errors == 0) begin
            $display("DOT PRODUCT UNIT TB: PASS -- %0d tests, 0 errors", num_tests);
            $finish;
        end else begin
            $fatal(1, "DOT PRODUCT UNIT TB: FAIL -- %0d tests, %0d errors",
                   num_tests, num_errors);
        end
    end

endmodule

`endif  // _DOT_PRODUCT_UNIT_TB_SV_



