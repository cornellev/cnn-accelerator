// ================================================================
//
// Date  : Oct 17, 2026
// Author: Ishaan Parikh
//
// Winograd kernel transform testbench.
//
// Expected U' values come from the same G' g G'^T as the Python
// export, so the hardware transform can replace stored U'.
//
// ================================================================

`ifndef _WINOGRAD_WEIGHT_TRANSFORM_TB_SV_
`define _WINOGRAD_WEIGHT_TRANSFORM_TB_SV_

module winograd_weight_transform_tb;

    logic signed  [7:0] weights   [0:8];
    // wire, not logic: Icarus 12 leaves an unpacked array output port
    // connected to a logic variable at X
    wire  signed [11:0] u_weights [0:15];

    winograd_weight_transform dut
    (
        .weights   (weights),
        .u_weights (u_weights)
    );

    integer num_vectors, num_errors;

    logic        [71:0]  weight_bits;
    logic        [191:0] expected_bits;
    logic signed [11:0]  expected_u [0:15];

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
                fields_read = $fscanf(file, "%b %b", weight_bits, expected_bits);

                if (fields_read == 2) begin
                    for (int i = 0; i < 9; i++) begin
                        weights[i] = weight_bits[71 - 8*i -: 8];
                    end
                    for (int i = 0; i < 16; i++) begin
                        expected_u[i] = expected_bits[191 - 12*i -: 12];
                    end

                    #1;
                    file_vectors = file_vectors + 1;
                    vector_error = 1'b0;
                    for (int i = 0; i < 16; i++) begin
                        if (u_weights[i] !== expected_u[i]) begin
                            vector_error = 1'b1;
                        end
                    end

                    if (vector_error) begin
                        file_errors = file_errors + 1;
                        if (file_errors <= 20)
                            $display("MISS weight_bits=%h | got u[0]=%0d u[5]=%0d u[10]=%0d u[15]=%0d | want %0d %0d %0d %0d",
                                     weight_bits,
                                     u_weights[0], u_weights[5], u_weights[10], u_weights[15],
                                     expected_u[0], expected_u[5], expected_u[10], expected_u[15]);
                    end
                end
            end
            $fclose(file);
            $display("%s: %0d vectors, %0d errors", path, file_vectors, file_errors);
            num_vectors = num_vectors + file_vectors;
            num_errors  = num_errors  + file_errors;
        end
    endtask

    // Waveforms: run vvp with +dump to write the first 50 vectors to
    // winograd_weight_transform_tb.vcd. Icarus skips unpacked arrays in
    // a plain $dumpvars, so every word is listed.
    initial begin
        if ($test$plusargs("dump")) begin
            $dumpfile("winograd_weight_transform_tb.vcd");
            for (int i = 0; i < 9; i++) begin
                $dumpvars(0, weights[i]);
            end
            for (int i = 0; i < 12; i++) begin
                $dumpvars(0, dut.t[i]);
            end
            for (int i = 0; i < 16; i++) begin
                $dumpvars(0, u_weights[i], expected_u[i]);
            end
            #50 $dumpoff;
        end
    end

    initial begin
        num_vectors = 0; num_errors = 0;
        run_vectors("tb/vectors/weight_transform_vectors.txt");
        if (num_errors == 0) begin
            $display("WINOGRAD_WEIGHT_TRANSFORM TB: PASS -- %0d vectors, 0 errors", num_vectors);
            $finish;
        end
        else begin
            $fatal(1, "WINOGRAD_WEIGHT_TRANSFORM TB: FAIL -- %0d vectors, %0d errors",
                   num_vectors, num_errors);
        end
    end

endmodule

`endif // _WINOGRAD_WEIGHT_TRANSFORM_TB_SV_
