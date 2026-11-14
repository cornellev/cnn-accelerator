// ================================================================
//
// Date  : Nov 7, 2026
// Author: Kaan Akan, Ishaan Parikh
//
// FPGA top-level for the MNIST feature extractor. Instantiates the
// mnist module with hardcoded picture inputs and an FSM that streams
// the picture into it one pixel per cycle.
//
// ================================================================

`ifndef _MNIST_FPGA_FSM_SV_
`define _MNIST_FPGA_FSM_SV_

`include "rtl/quant_params.sv"

module mnist_fpga_fsm (
    input  clk,
    input  rst_n,

    output       output_valid,
    output [3:0] num_output
);

    // Load input image pixels as a 1D array
    logic signed [ACTIVATION_BITS-1:0] input_pixels [0:783];

    initial begin
        $readmemh("rtl/mem/mnist-4.mem", input_pixels);
    end

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_MNIST_BUSY,
        ST_CONFIDENCE_CALC,
        ST_DONE
    } state_t;

    state_t current_state, next_state;

    // State register
    always_ff @(posedge clk) begin : state_register
        if (!rst_n) begin
            current_state <= ST_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    localparam int LAYER_1_IN_CHANS  = 1;
    localparam int LAYER_1_OUT_CHANS = 8;
    localparam int LAYER_3_IN_CHANS  = LAYER_1_OUT_CHANS;
    localparam int LAYER_3_OUT_CHANS = 16;
    localparam int LAYER_4_OUT_CHANS = LAYER_3_OUT_CHANS;

    // Layout from model/mnist_cnn_int8.json: each kernel is its weights
    // followed by a 4-byte bias, then the layer's scale multipliers,
    // x_zero and y_zero come after all of its kernels.
    localparam int LAYER_1_OFFSET        = 0;
    localparam int LAYER_1_KERNEL_STRIDE = LAYER_1_IN_CHANS * 9 + 4;
    localparam int LAYER_1_SCALE_OFFSET  = LAYER_1_OFFSET + LAYER_1_OUT_CHANS * LAYER_1_KERNEL_STRIDE;  // 104
    localparam int LAYER_1_X_ZERO_OFFSET = LAYER_1_SCALE_OFFSET + LAYER_1_OUT_CHANS * 4;                // 136
    localparam int LAYER_1_Y_ZERO_OFFSET = LAYER_1_X_ZERO_OFFSET + 1;                                   // 137

    localparam int LAYER_3_OFFSET        = LAYER_1_Y_ZERO_OFFSET + 1;                                   // 138
    localparam int LAYER_3_KERNEL_STRIDE = LAYER_3_IN_CHANS * 9 + 4;
    localparam int LAYER_3_SCALE_OFFSET  = LAYER_3_OFFSET + LAYER_3_OUT_CHANS * LAYER_3_KERNEL_STRIDE;  // 1354
    localparam int LAYER_3_Y_ZERO_OFFSET = LAYER_3_SCALE_OFFSET + LAYER_3_OUT_CHANS * 4 + 1;    
    
    localparam  int NUM_PIXELS = 28 * 28;        // 1419

    // Quantized model parameters, one hex byte per line. Weights are int8,
    // biases and scale multipliers are little-endian int32.
    localparam int MODEL_BYTES = 9341;

    logic [7:0] model_rom [0:MODEL_BYTES-1];

    initial begin
        $readmemh("model/mnist_cnn_int8.mem", model_rom);
    end

    function automatic logic signed [31:0] read_int32(int offset);
        return {model_rom[offset+3], model_rom[offset+2], model_rom[offset+1], model_rom[offset]};
    endfunction

    logic signed [    WEIGHT_BITS-1:0] layer_1_weights           [0:LAYER_1_OUT_CHANS-1][0:LAYER_1_IN_CHANS-1][0:8];
    logic signed [      BIAS_BITS-1:0] layer_1_bias              [0:LAYER_1_OUT_CHANS-1];
    logic        [   SCALE_M_BITS-1:0] layer_1_quant_scale_mult  [0:LAYER_1_OUT_CHANS-1];
    logic signed [ACTIVATION_BITS-1:0] layer_1_quant_offset;
    logic signed [ACTIVATION_BITS-1:0] layer_1_prev_quant_offset;

    logic signed [    WEIGHT_BITS-1:0] layer_3_weights           [0:LAYER_3_OUT_CHANS-1][0:LAYER_3_IN_CHANS-1][0:8];
    logic signed [      BIAS_BITS-1:0] layer_3_bias              [0:LAYER_3_OUT_CHANS-1];
    logic        [   SCALE_M_BITS-1:0] layer_3_quant_scale_mult  [0:LAYER_3_OUT_CHANS-1];
    logic signed [ACTIVATION_BITS-1:0] layer_3_quant_offset;

    // Unpack the model bytes into the mnist ports
    always_comb begin
        for (int k = 0; k < LAYER_1_OUT_CHANS; k++) begin
            for (int c = 0; c < LAYER_1_IN_CHANS; c++) begin
                for (int i = 0; i < 9; i++) begin
                    layer_1_weights[k][c][i] = model_rom[LAYER_1_OFFSET + k * LAYER_1_KERNEL_STRIDE + c * 9 + i];
                end
            end
            layer_1_bias[k]             = BIAS_BITS'(read_int32(LAYER_1_OFFSET + k * LAYER_1_KERNEL_STRIDE + LAYER_1_IN_CHANS * 9));
            layer_1_quant_scale_mult[k] = SCALE_M_BITS'(read_int32(LAYER_1_SCALE_OFFSET + k * 4));
        end
        layer_1_quant_offset      = model_rom[LAYER_1_Y_ZERO_OFFSET];
        layer_1_prev_quant_offset = model_rom[LAYER_1_X_ZERO_OFFSET];

        for (int k = 0; k < LAYER_3_OUT_CHANS; k++) begin
            for (int c = 0; c < LAYER_3_IN_CHANS; c++) begin
                for (int i = 0; i < 9; i++) begin
                    layer_3_weights[k][c][i] = model_rom[LAYER_3_OFFSET + k * LAYER_3_KERNEL_STRIDE + c * 9 + i];
                end
            end
            layer_3_bias[k]             = BIAS_BITS'(read_int32(LAYER_3_OFFSET + k * LAYER_3_KERNEL_STRIDE + LAYER_3_IN_CHANS * 9));
            layer_3_quant_scale_mult[k] = SCALE_M_BITS'(read_int32(LAYER_3_SCALE_OFFSET + k * 4));
        end
        layer_3_quant_offset = model_rom[LAYER_3_Y_ZERO_OFFSET];
    end

    // Read classifier parameters from memory file
    localparam int NUM_CLASSES              = 10;
    localparam int CLASSIFIER_IN_FEATURES   = LAYER_4_OUT_CHANS * 7 * 7;
    localparam int CLASSIFIER_OFFSET        = LAYER_3_Y_ZERO_OFFSET + 1;                                  // 1420
    localparam int CLASSIFIER_KERNEL_STRIDE = CLASSIFIER_IN_FEATURES + 4;
    localparam int CLASSIFIER_SCALE_OFFSET  = CLASSIFIER_OFFSET + NUM_CLASSES * CLASSIFIER_KERNEL_STRIDE;  // 9300
    localparam int CLASSIFIER_Y_ZERO_OFFSET = CLASSIFIER_SCALE_OFFSET + NUM_CLASSES * 4;                   // 9340

    logic signed [    WEIGHT_BITS-1:0] classifier_weights          [0:NUM_CLASSES-1][0:CLASSIFIER_IN_FEATURES-1];
    logic signed [      BIAS_BITS-1:0] classifier_bias             [0:NUM_CLASSES-1];
    logic        [   SCALE_M_BITS-1:0] classifier_quant_scale_mult [0:NUM_CLASSES-1];
    logic signed [ACTIVATION_BITS-1:0] classifier_quant_offset;

    always_comb begin
        for (int k = 0; k < NUM_CLASSES; k++) begin
            for (int i = 0; i < CLASSIFIER_IN_FEATURES; i++) begin
                classifier_weights[k][i] = model_rom[CLASSIFIER_OFFSET + k * CLASSIFIER_KERNEL_STRIDE + i];
            end
            classifier_bias[k]             = BIAS_BITS'(read_int32(CLASSIFIER_OFFSET + k * CLASSIFIER_KERNEL_STRIDE + CLASSIFIER_IN_FEATURES));
            classifier_quant_scale_mult[k] = SCALE_M_BITS'(read_int32(CLASSIFIER_SCALE_OFFSET + k * 4));
        end
        classifier_quant_offset = model_rom[CLASSIFIER_Y_ZERO_OFFSET];
    end

    logic                              mnist_start;
    logic                              mnist_pixel_valid;
    logic signed [ACTIVATION_BITS-1:0] mnist_input_pixel  [0:LAYER_1_IN_CHANS-1];
    logic                              mnist_done;
    logic                              mnist_output_valid;
    logic signed [ACTIVATION_BITS-1:0] mnist_output_pixel [0:LAYER_4_OUT_CHANS-1];

    mnist mnist_model (
        .clk                       (clk),
        .rst_n                     (rst_n),
        .start                     (mnist_start),

        .pixel_valid               (mnist_pixel_valid),
        .input_pixel               (mnist_input_pixel),

        .layer_1_weights           (layer_1_weights),
        .layer_1_bias              (layer_1_bias),
        .layer_1_quant_scale_mult  (layer_1_quant_scale_mult),
        .layer_1_quant_offset      (layer_1_quant_offset),
        .layer_1_prev_quant_offset (layer_1_prev_quant_offset),

        .layer_3_weights           (layer_3_weights),
        .layer_3_bias              (layer_3_bias),
        .layer_3_quant_scale_mult  (layer_3_quant_scale_mult),
        .layer_3_quant_offset      (layer_3_quant_offset),

        .done                      (mnist_done),
        .output_valid              (mnist_output_valid),
        .output_pixel              (mnist_output_pixel)
    );

    logic classifier_done;

    always_comb begin : next_state_logic
        next_state = current_state;
        case (current_state)
            ST_IDLE: begin
                next_state = ST_MNIST_BUSY;
            end

            ST_MNIST_BUSY: begin
                if (mnist_done) begin
                    next_state = ST_CONFIDENCE_CALC;
                end
            end

            ST_CONFIDENCE_CALC: begin
                if (classifier_done) begin
                    next_state = ST_DONE;
                end
            end

            ST_DONE: begin
                next_state = ST_DONE;
            end

            default: next_state = ST_IDLE;
        endcase
    end

    logic signed [ACTIVATION_BITS-1:0] flattened_last_layer [0:783];
    logic [9:0] output_count;

    logic [$clog2(NUM_PIXELS+1)-1:0] pixel_count;

    always_ff @(posedge clk) begin : pixel_counter
     if (!rst_n || current_state == ST_IDLE) begin
        pixel_count <= '0;

     end 
     else if (mnist_pixel_valid) begin
        pixel_count <= pixel_count + 1'b1;
     end
    end

    assign mnist_start          = (current_state == ST_IDLE);
    assign mnist_pixel_valid    = (current_state == ST_MNIST_BUSY) && (pixel_count < NUM_PIXELS);
    assign mnist_input_pixel[0] = input_pixels[pixel_count];
    assign output_valid         = (current_state == ST_DONE);

    always_ff @(posedge clk) begin
        if (current_state == ST_IDLE || !rst_n) begin
            for (int i = 0; i < 784; i++) begin
                flattened_last_layer[i] <= '0;
            end
            output_count <= '0;
        end
        else if (mnist_output_valid) begin
            output_count <= output_count + 1'b1;
            for (int j = 0; j < 16; j++) begin
                flattened_last_layer[32'(output_count) + 49*j] <= mnist_output_pixel[j];
            end
        end
    end

    // Confidence calculation classifier
    logic [3:0] classifier_num_output;

    confidence_classifier #(
        .N           (7 * 7),
        .ROW_WIDTH   (CLASSIFIER_IN_FEATURES),
        .NUM_CLASSES (NUM_CLASSES)
    ) classifier (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .in_classification_state (current_state == ST_CONFIDENCE_CALC),
        .flattened_last_layer    (flattened_last_layer),
        .weights                 (classifier_weights),
        .bias                    (classifier_bias),
        .quant_scale_mult        (classifier_quant_scale_mult),
        .num_valid               (classifier_done),
        .num_output              (classifier_num_output)
    );

    // Hold the digit after the classifier's one-cycle valid
    logic [3:0] reg_num_output;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            reg_num_output <= '0;
        end
        else if (classifier_done) begin
            reg_num_output <= classifier_num_output;
        end
    end

    assign num_output = reg_num_output;

endmodule

`endif  // _MNIST_FPGA_FSM_SV_
