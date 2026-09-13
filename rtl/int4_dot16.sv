// Combinational 16-lane signed INT8 x signed INT4 dot-product primitive.
//
// Lane packing (least-significant lane first):
//   q_flat[i*8 +: 8] = q[i]
//   k_flat[i*4 +: 4] = k[i]
//
// Product widths and tree widths are sized for every representable input:
// 12-bit products, then 13/14/15/16-bit pairwise reductions.
module int4_dot16 (
    input  logic [127:0] q_flat,
    input  logic [63:0]  k_flat,
    output logic signed [15:0] dot
);
    logic signed [11:0] product [0:15];
    logic signed [12:0] sum2    [0:7];
    logic signed [13:0] sum4    [0:3];
    logic signed [14:0] sum8    [0:1];

    genvar lane;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : gen_products
            assign product[lane] = $signed(q_flat[lane * 8 +: 8])
                                 * $signed(k_flat[lane * 4 +: 4]);
        end
        for (lane = 0; lane < 8; lane = lane + 1) begin : gen_sum2
            assign sum2[lane] = $signed({product[lane * 2][11], product[lane * 2]})
                              + $signed({product[lane * 2 + 1][11], product[lane * 2 + 1]});
        end
        for (lane = 0; lane < 4; lane = lane + 1) begin : gen_sum4
            assign sum4[lane] = $signed({sum2[lane * 2][12], sum2[lane * 2]})
                              + $signed({sum2[lane * 2 + 1][12], sum2[lane * 2 + 1]});
        end
        for (lane = 0; lane < 2; lane = lane + 1) begin : gen_sum8
            assign sum8[lane] = $signed({sum4[lane * 2][13], sum4[lane * 2]})
                              + $signed({sum4[lane * 2 + 1][13], sum4[lane * 2 + 1]});
        end
    endgenerate

    assign dot = $signed({sum8[0][14], sum8[0]}) + $signed({sum8[1][14], sum8[1]});
endmodule
