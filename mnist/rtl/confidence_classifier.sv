// ================================================================
//
// Date  : Nov 14, 2026
// Author: Kaan Akan
//
// Confidence classifer for MNIST. Takes the flattened output of the
// last layer and finds the corresponding output value that is the greatest.
//
// Multiplication values are loaded in from mnist_cnn_int8.mem in
// the FSM, and is passed from higher level with ports.
// ================================================================

`ifndef _CONFIDENCE_CLASSIFIER_SV_
`define _CONFIDENCE_CLASSIFIER_SV_

`include "rtl/quant_params.sv"

module confidence_classifier #(
    parameter int N           = 49,   // elements per dot product
    parameter int ROW_WIDTH   = 784,  // features per class (16 * 7 * 7)
    parameter int NUM_CLASSES = 10
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                              in_classification_state,
    input  logic signed [ACTIVATION_BITS-1:0] flattened_last_layer [0:783],

    input  logic signed [WEIGHT_BITS-1:0]  weights          [0:9][0:783],
    input  logic signed [BIAS_BITS-1:0]    bias             [0:9],
    input  logic        [SCALE_M_BITS-1:0] quant_scale_mult [0:9],

    output logic       num_valid,
    output logic [3:0] num_output
);
    localparam int NUM_CHUNKS = ROW_WIDTH / N;

    logic signed [ACTIVATION_BITS-1:0] current_activations [0:N-1];
    logic signed [WEIGHT_BITS-1:0] current_weights [0:NUM_CLASSES-1][0:N-1];

    logic signed [ACTIVATION_BITS+WEIGHT_BITS+$clog2(ROW_WIDTH)-1:0] current_dot_products [0:9];

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_DOT_PRODUCT,
        ST_ADDITION,
        ST_COMPARISON,
        ST_OUTPUT_VALID
    } state_t;

    // State Register
    state_t current_state, next_state;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            current_state <= ST_IDLE;
        end
        else begin
            current_state <= next_state;
        end
    end

    // Next State Logic
    always_comb begin : next_state_logic
        next_state = current_state;
        case (current_state)
            ST_IDLE: begin
                if (in_classification_state) begin
                    next_state = ST_DOT_PRODUCT;
                end
            end
            ST_DOT_PRODUCT: begin
                if ((dot_product_cycle_count == 4'(NUM_CHUNKS - 1))) begin
                    next_state = ST_ADDITION;
                end
            end
            ST_ADDITION: begin
                next_state = ST_COMPARISON;
            end
            ST_COMPARISON: begin
                next_state = ST_OUTPUT_VALID;
            end
            ST_OUTPUT_VALID: begin
                next_state = ST_IDLE;
            end

            default: next_state = ST_IDLE;
        endcase
    end

    // Output Logic
    always_comb begin : output_logic
        num_output = '0;
        num_valid  = '0;

        case (current_state)
            ST_OUTPUT_VALID: begin
                num_output = best_class;
                num_valid  = 1'b1;
            end
            default: ;
        endcase
    end

    // Dot product cycle counter and result register
    logic [$clog2(NUM_CHUNKS)-1:0] dot_product_cycle_count;
    logic signed [ACTIVATION_BITS+WEIGHT_BITS+$clog2(ROW_WIDTH)-1:0] intermediate_results [0:NUM_CLASSES-1][0:NUM_CHUNKS-1];

    always_ff @(posedge clk) begin
        if (!rst_n || (current_state == ST_IDLE)) begin
            dot_product_cycle_count <= '0;
            intermediate_results <= '{default: '0};
        end
        else if (current_state == ST_DOT_PRODUCT) begin
            dot_product_cycle_count <= dot_product_cycle_count + 1'b1;
            for (int k = 0; k < NUM_CLASSES; k++) begin
                intermediate_results[k][dot_product_cycle_count] <= current_dot_products[k];
            end
        end
    end

    // Select chunk dot_product_cycle_count of the activations and each class's weights
    always_comb begin
        for (int i = 0; i < N; i++) begin
            current_activations[i] = flattened_last_layer[dot_product_cycle_count*N + i];
            for (int k = 0; k < NUM_CLASSES; k++) begin
                current_weights[k][i] = weights[k][dot_product_cycle_count*N + i];
            end
        end
    end

    // Dot product units, in which one unit does the weight for each digit
    for (genvar i = 0; i < (NUM_CLASSES); i++) begin :generate_dot_product_units
        dot_product_unit #(
            .N         (N),
            .ROW_WIDTH (ROW_WIDTH)
        ) dot_product_engine 
        (
            .activations (current_activations),
            .weights     (current_weights[i]),
            .dot_product (current_dot_products[i])
        );
    end

    // Add intermediate dot products in binary tree and register result
    localparam int SUM_BITS   = ACTIVATION_BITS + WEIGHT_BITS + $clog2(ROW_WIDTH) + $clog2(NUM_CHUNKS);
    localparam int SCORE_BITS = (SUM_BITS + 1) + (SCALE_M_BITS + 1);

    logic signed [SUM_BITS-1:0] summed_results     [0:NUM_CLASSES-1];
    logic signed [SUM_BITS:0]   reg_summed_results [0:NUM_CLASSES-1];  // add one extra bit for the bias

    for (genvar k = 0; k < NUM_CLASSES; k++) begin : generate_adders
        binary_tree_adder #(
            .N       (NUM_CHUNKS),
            .IN_BITS (ACTIVATION_BITS + WEIGHT_BITS + $clog2(ROW_WIDTH))
        ) binary_adder_units
        (
            .addends (intermediate_results[k]),
            .sum     (summed_results[k])
        );
    end

    always_ff @(posedge clk) begin
        if (!rst_n || (current_state == ST_IDLE)) begin
            reg_summed_results <= '{default: '0};
        end
        else if (current_state == ST_ADDITION) begin
            for (int k = 0; k < NUM_CLASSES; k++) begin
                reg_summed_results[k] <= summed_results[k] + (SUM_BITS+1)'(bias[k]);
            end
        end
    end

    // Scale each class's sum so they can be compared
    logic signed [SCORE_BITS-1:0] scores [0:NUM_CLASSES-1];

    always_comb begin
        for (int k = 0; k < NUM_CLASSES; k++) begin
            scores[k] = reg_summed_results[k] * $signed({1'b0, quant_scale_mult[k]});
        end
    end

    // Comparison logic
    logic [3:0] best_class;

    always_comb begin
        best_class = '0;
        for (int k = 1; k < NUM_CLASSES; k++) begin
            if (scores[k] > scores[best_class]) begin
                best_class = 4'(k);
            end
        end
    end

endmodule

`endif  // _CONFIDENCE_CLASSIFIER_SV_
