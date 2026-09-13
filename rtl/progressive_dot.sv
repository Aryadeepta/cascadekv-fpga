// Combinational progressive Q8 x K4 group-score update.
//
// scale_product is signed Q4.12.  int4_dot16 returns an integer, so their
// exact 32-bit product has 12 fractional bits.  The Q11.13 score domain has
// one more fractional bit; its contribution code is therefore product << 1.
// Both the contribution and the following accumulator sum saturate rather
// than wrap when their mathematical Q11.13 codes exceed 24-bit signed range.
module progressive_dot (
    input  logic [127:0] q_flat,
    input  logic [63:0]  k_flat,
    input  logic signed [15:0] scale_product,
    input  logic signed [23:0] score_in,
    output logic signed [15:0] integer_dot,
    output logic signed [23:0] contribution,
    output logic signed [23:0] score_out,
    output logic               contribution_saturated,
    output logic               score_saturated
);
    // A signed 16 x signed 16 multiply is exactly 32 bits.  The extra bit in
    // aligned_contribution preserves the explicit Q4.12 -> Q11.13 left shift.
    logic signed [31:0] raw_product;
    logic signed [32:0] aligned_contribution;
    logic signed [24:0] score_sum;

    localparam logic signed [23:0] SCORE_MAX = 24'sh7fffff;
    localparam logic signed [23:0] SCORE_MIN = 24'sh800000;

    int4_dot16 dot16 (
        .q_flat(q_flat),
        .k_flat(k_flat),
        .dot(integer_dot)
    );

    assign raw_product = $signed(integer_dot) * $signed(scale_product);
    assign aligned_contribution = $signed({raw_product[31], raw_product}) <<< 1;

    always_comb begin
        contribution_saturated = 1'b0;
        if (aligned_contribution > $signed({{9{SCORE_MAX[23]}}, SCORE_MAX})) begin
            contribution = SCORE_MAX;
            contribution_saturated = 1'b1;
        end else if (aligned_contribution < $signed({{9{SCORE_MIN[23]}}, SCORE_MIN})) begin
            contribution = SCORE_MIN;
            contribution_saturated = 1'b1;
        end else begin
            contribution = aligned_contribution[23:0];
        end

        // Sign-extend each Q11.13 code before adding in a 25-bit domain.
        score_sum = $signed({score_in[23], score_in})
                  + $signed({contribution[23], contribution});
        score_saturated = 1'b0;
        if (score_sum > $signed({SCORE_MAX[23], SCORE_MAX})) begin
            score_out = SCORE_MAX;
            score_saturated = 1'b1;
        end else if (score_sum < $signed({SCORE_MIN[23], SCORE_MIN})) begin
            score_out = SCORE_MIN;
            score_saturated = 1'b1;
        end else begin
            score_out = score_sum[23:0];
        end
    end
endmodule
