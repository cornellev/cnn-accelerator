// ================================================================
//
// Date  : Oct 6, 2026
// Author: Ishaan Parikh
//
// Winograd 3 x 3 convolution module testbench.
//
// Expected outputs are computed with direct convolution, so the
// Winograd transforms are verified for actual use. No innacuracy.
//
// ================================================================

`ifndef _WINOGRAD_3X3_CONV_TB_SV_
`define _WINOGRAD_3X3_CONV_TB_SV_

module winograd_3x3_conv_tb;

    logic         [7:0] tile        [0:15];
    logic signed [11:0] u_weights   [0:15];
    // wire, not logic: Icarus 12 leaves an unpacked array output port
    // connected to a logic variable at X
    wire  signed [19:0] convolution [0:3];

    winograd_3x3_conv dut
    (
        .tile        (tile),
        .u_weights   (u_weights),
        .convolution (convolution)
    );

    integer num_vectors, num_errors;

    logic        [127:0] tile_bits;
    logic        [191:0] u_bits;
    logic        [79:0]  expected_bits;
    logic signed [19:0]  expected_convolution [0:3];

    task run_vectors(input string path);
        integer file, fields_read;
        integer file_vectors, file_errors;
        logic   vector_error;
        begin
            file_vectors = 0; file_errors = 0;
            file = $fopen(path, "r");
            if (file == 0) begin
                $fatal(1, "ERROR: cannot open %s", path);
            end
            while (!$feof(file)) begin
                fields_read = $fscanf(file, "%b %b %b", tile_bits, u_bits, expected_bits);

                if (fields_read == 3) begin
                    for (int i = 0; i < 16; i++) begin
                        tile[i]      = tile_bits[127 - 8*i -: 8];
                        u_weights[i] = u_bits[191 - 12*i -: 12];
                    end
                    for (int i = 0; i < 4; i++) begin
                        expected_convolution[i] = expected_bits[79 - 20*i -: 20];
                    end

                    #1;
                    file_vectors = file_vectors + 1;
                    vector_error = 1'b0;
                    for (int i = 0; i < 4; i++) begin
                        if (convolution[i] !== expected_convolution[i]) begin
                            vector_error = 1'b1;
                        end
                    end

                    if (vector_error) begin
                        file_errors = file_errors + 1;
                        if (file_errors <= 20)
                            $display("MISS tile_bits=%h u_bits=%h | got %0d %0d %0d %0d | want %0d %0d %0d %0d",
                                     tile_bits, u_bits,
                                     convolution[0], convolution[1], convolution[2], convolution[3],
                                     expected_convolution[0], expected_convolution[1],
                                     expected_convolution[2], expected_convolution[3]);
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
        run_vectors("tb/vectors/winograd_vectors.txt");
        if (num_errors == 0) begin
            $display("WINOGRAD_3x3_CONV TB: PASS -- %0d vectors, 0 errors", num_vectors);
            $finish;
        end
        else begin
            $fatal(1, "WINOGRAD_3x3_CONV TB: FAIL -- %0d vectors, %0d errors",
                   num_vectors, num_errors);
        end
    end

endmodule

`endif // _WINOGRAD_3X3_CONV_TB_SV_
