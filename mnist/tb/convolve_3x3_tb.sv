// ================================================================
//
// Date  : Sep 19, 2026
// Author: Kaan Akan
//
// 3 x 3 convolution module testbench.
//
// ================================================================

`ifndef _CONVOLVE_3X3_TB_SV_
`define _CONVOLVE_3X3_TB_SV_

module convolve_3x3_tb;

    logic         [7:0] pixels  [0:8];
    logic signed  [7:0] weights [0:8];
    logic signed [19:0] convolution;

    convolve_3x3 dut
    (
        .pixels      (pixels),
        .weights     (weights),
        .convolution (convolution)
    );

    integer num_vectors, num_errors;

    logic        [71:0] pixel_bits, weight_bits;
    logic signed [19:0] expected_convolution;

    task run_vectors(input string path);
        integer file, fields_read;
        integer file_vectors, file_errors;
        begin
            file_vectors = 0; file_errors = 0;
            file = $fopen(path, "r");
            if (file == 0) begin
                $fatal(1, "ERROR: cannot open %s", path);
            end
            while (!$feof(file)) begin
                fields_read = $fscanf(file, "%b %b %b", pixel_bits, weight_bits, expected_convolution);

                for (int i = 0; i < 9; i++) begin
                    pixels[i]  = pixel_bits[71 - 8*i -: 8];
                    weights[i] = weight_bits[71 - 8*i -: 8];
                end
                
                if (fields_read == 3) begin
                    #1;
                    file_vectors = file_vectors + 1;
                    if ((convolution !== expected_convolution)) begin
                        file_errors = file_errors + 1;
                        if (file_errors <= 20)
                            $display("MISS pixel_bits=%b weight_bits=%b | got convolution=%b | want %b",
                                     pixel_bits, weight_bits, convolution, expected_convolution);
                    end
                end
            end
            $fclose(file);
            $display("%s: %0d vectors, %0d errors", path, file_vectors, file_errors);
            num_vectors = num_vectors + file_vectors;
            num_errors  = num_errors  + file_errors;
        end
    endtask

    initial begin
        num_vectors = 0; num_errors = 0;
        run_vectors("tb/vectors/convolve_vectors.txt");
        if (num_errors == 0) begin
            $display("CONVOLVE_3x3 TB: PASS -- %0d vectors, 0 errors", num_vectors);
            $finish;
        end
        else begin
            $fatal(1, "CONVOLVE_3x3 TB TB: FAIL -- %0d vectors, %0d errors",
                   num_vectors, num_errors);
        end
    end

endmodule

`endif // _CONVOLVE_3X3_TB_SV_