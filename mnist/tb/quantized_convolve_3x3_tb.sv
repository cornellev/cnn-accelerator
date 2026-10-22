// ================================================================
//
// Date  : Oct 6, 2026
// Author: Kaan Akan
//
// 3 x 3 quantized convolution module testbench.
//
// ================================================================

`ifndef _QUANTIZED_CONVOLVE_3X3_TB_SV_
`define _QUANTIZED_CONVOLVE_3X3_TB_SV_

`include "rtl/quant_params.sv"

module quantized_convolve_3x3_tb;

    logic signed [ACTIVATION_BITS-1:0] pixels [0:8];
    logic signed [    WEIGHT_BITS-1:0] weights[0:8];
    logic signed [ACTIVATION_BITS-1:0] quantized_convolution;
    logic signed [      BIAS_BITS-1:0] full_bias;
    logic        [   SCALE_M_BITS-1:0] quant_scale_mult;
    logic signed [ACTIVATION_BITS-1:0] quant_offset;

    quantized_convolution dut (
        .pixels          (pixels),
        .weights         (weights),
        .full_bias       (full_bias),
        .quant_scale_mult(quant_scale_mult),
        .quant_offset    (quant_offset),
        .quantized_value (quantized_convolution)
    );

    integer num_vectors, num_errors;

    logic [9*ACTIVATION_BITS-1:0] pixel_bits;
    logic [9*WEIGHT_BITS-1:0] weight_bits;
    logic signed [ACTIVATION_BITS-1:0] expected_convolution;

    task run_vectors(input string path);
        integer file, fields_read;
        integer file_vectors, file_errors;
        begin
            file_vectors = 0;
            file_errors = 0;
            file = $fopen(path, "r");
            if (file == 0) begin
                $fatal(1, "ERROR: cannot open %s", path);
            end
            while (!$feof(file)) begin
                fields_read = $fscanf(
                    file, "%b %b %b %b %b %b",
                    pixel_bits, weight_bits, full_bias,
                    quant_scale_mult, quant_offset, expected_convolution
                );

                for (int i = 0; i < 9; i++) begin
                    pixels[i]  = pixel_bits[(9-i)*ACTIVATION_BITS-1-:ACTIVATION_BITS];
                    weights[i] = weight_bits[(9-i)*WEIGHT_BITS-1-:WEIGHT_BITS];
                end

                if (fields_read == 6) begin
                    #1;
                    file_vectors = file_vectors + 1;
                    if ((quantized_convolution !== expected_convolution)) begin
                        file_errors = file_errors + 1;
                        if (file_errors <= 20)
                            $display(
                                "MISS pixel_bits=%b weight_bits=%b | got convolution=%b | want %b",
                                pixel_bits, weight_bits,
                                quantized_convolution, expected_convolution
                            );
                    end
                end
            end
            $fclose(file);
            $display("%s: %0d vectors, %0d errors", path, file_vectors, file_errors);
            num_vectors = num_vectors + file_vectors;
            num_errors  = num_errors + file_errors;
        end
    endtask

    initial begin
        num_vectors = 0;
        num_errors = 0;
        run_vectors("tb/vectors/quantized_convolve_vectors.txt");
        if (num_errors == 0) begin
            $display("QUANTIZED CONVOLUTION TB: PASS -- %0d vectors, 0 errors", num_vectors);
            $finish;
        end else begin
            $fatal(1, "QUANTIZED CONVOLUTION TB: FAIL -- %0d vectors, %0d errors",
                   num_vectors, num_errors);
        end
    end

endmodule

`endif  // _QUANTIZED_CONVOLVE_3X3_TB_SV_
