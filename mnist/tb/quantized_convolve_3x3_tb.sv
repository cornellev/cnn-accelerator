// 3x3 quantized convolution: compare against quantize_custom.py vectors.
`ifndef _QUANTIZED_CONVOLVE_3X3_TB_SV_
`define _QUANTIZED_CONVOLVE_3X3_TB_SV_

`include "rtl/quant_params.sv"

module quantized_convolve_3x3_tb;
    logic signed [ACTIVATION_BITS-1:0] pixels [0:8];
    logic signed [WEIGHT_BITS-1:0] weights [0:8];
    logic signed [BIAS_BITS-1:0] full_bias;
    logic [SCALE_M_BITS-1:0] quant_scale_mult;
    logic signed [ACTIVATION_BITS-1:0] quant_offset, quantized_value, expected;
    logic [9*ACTIVATION_BITS-1:0] pixel_bits;
    logic [9*WEIGHT_BITS-1:0] weight_bits;
    integer file, fields_read, num_vectors, num_errors;

    quantized_convolution dut (
        .pixels(pixels),
        .weights(weights),
        .full_bias(full_bias),
        .quant_scale_mult(quant_scale_mult),
        .quant_offset(quant_offset),
        .quantized_value(quantized_value)
    );

    initial begin
        num_vectors = 0;
        num_errors = 0;
        file = $fopen("tb/vectors/quantized_convolve_vectors.txt", "r");
        if (file == 0) $fatal(1, "Cannot open quantized convolution vectors");
        while (!$feof(file)) begin
            fields_read = $fscanf(file, "%b %b %b %b %b %b", pixel_bits, weight_bits,
                                  full_bias, quant_scale_mult, quant_offset, expected);
            if (fields_read == 6) begin
                for (int i = 0; i < 9; i++) begin
                    pixels[i] = pixel_bits[(9-i)*ACTIVATION_BITS-1-:ACTIVATION_BITS];
                    weights[i] = weight_bits[(9-i)*WEIGHT_BITS-1-:WEIGHT_BITS];
                end
                #1;
                num_vectors++;
                if (quantized_value !== expected) begin
                    num_errors++;
                    if (num_errors <= 20)
                        $display("MISS vector %0d got=%0d want=%0d", num_vectors, quantized_value, expected);
                end
            end else if (fields_read != -1) begin
                $fatal(1, "Malformed quantized convolution vector after %0d vectors", num_vectors);
            end
        end
        $fclose(file);
        if (num_vectors == 0 || num_errors != 0)
            $fatal(1, "QUANTIZED CONVOLUTION TB: FAIL -- %0d vectors, %0d errors", num_vectors, num_errors);
        $display("QUANTIZED CONVOLUTION TB: PASS -- %0d vectors, 0 errors", num_vectors);
        $finish;
    end
endmodule

`endif // _QUANTIZED_CONVOLVE_3X3_TB_SV_
