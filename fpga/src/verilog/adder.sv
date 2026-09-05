module adder #(
    parameter N
) (
    input wire clk,
    input wire [N-1:0] a,
    input wire [N-1:0] b,
    output wire [N-1:0] c
);
    assign c = a + b;
endmodule

