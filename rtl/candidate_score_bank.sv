// Persistent per-candidate Q11.13 partial-score storage.
//
// Requests are accepted only while idle (update_valid is sampled in IDLE).
// A request accepted on clock edge N produces result_valid on edge N+2:
//   N:   register request and synchronously read the previous score
//   N+1: register the combinational progressive_dot result
//   N+2: write the result and present it at the result interface
//
// reset deliberately resets only control/pipeline state.  score_mem is not
// reset so that FPGA tools may infer it as a memory; stage-16 callers use
// clear_before_update to supply a zero previous score.
module candidate_score_bank #(
    parameter integer CANDIDATE_COUNT = 256,
    parameter integer ID_WIDTH = $clog2(CANDIDATE_COUNT),
    parameter integer SCORE_WIDTH = 24
) (
    input  logic                        clk,
    input  logic                        rst,
    input  logic                        update_valid,
    output logic                        update_ready,
    input  logic [ID_WIDTH-1:0]         candidate_id,
    input  logic [127:0]                q_flat,
    input  logic [63:0]                 k_flat,
    input  logic signed [15:0]          scale_product,
    input  logic                        clear_before_update,
    output logic                        result_valid,
    output logic [ID_WIDTH-1:0]         result_candidate_id,
    output logic signed [SCORE_WIDTH-1:0] result_score,
    output logic                        contribution_saturated,
    output logic                        score_saturated
);
    typedef enum logic [1:0] {IDLE, COMPUTE, WRITE_RESULT} state_t;
    state_t state;

    // The bank has one request slot.  Keeping this separate from the
    // registered result interface makes request acceptance unambiguous.
    assign update_ready = (state == IDLE);

    // SCORE_WIDTH is 24 for the frozen progressive_dot Q11.13 interface.
    logic signed [SCORE_WIDTH-1:0] score_mem [0:CANDIDATE_COUNT-1];
    logic [ID_WIDTH-1:0] request_id;
    logic [127:0] request_q_flat;
    logic [63:0] request_k_flat;
    logic signed [15:0] request_scale_product;
    logic signed [SCORE_WIDTH-1:0] previous_score;
    logic signed [SCORE_WIDTH-1:0] pending_score;
    logic pending_contribution_saturated;
    logic pending_score_saturated;

    logic signed [23:0] dot_score_out;
    logic dot_contribution_saturated;
    logic dot_score_saturated;
    logic signed [15:0] unused_integer_dot;
    logic signed [23:0] unused_contribution;

    progressive_dot update_dot (
        .q_flat(request_q_flat),
        .k_flat(request_k_flat),
        .scale_product(request_scale_product),
        .score_in(previous_score),
        .integer_dot(unused_integer_dot),
        .contribution(unused_contribution),
        .score_out(dot_score_out),
        .contribution_saturated(dot_contribution_saturated),
        .score_saturated(dot_score_saturated)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            result_valid <= 1'b0;
            result_candidate_id <= '0;
            result_score <= '0;
            contribution_saturated <= 1'b0;
            score_saturated <= 1'b0;
            request_id <= '0;
            request_q_flat <= '0;
            request_k_flat <= '0;
            request_scale_product <= '0;
            previous_score <= '0;
            pending_score <= '0;
            pending_contribution_saturated <= 1'b0;
            pending_score_saturated <= 1'b0;
        end else begin
            result_valid <= 1'b0;
            case (state)
                IDLE: begin
                    if (update_valid && update_ready) begin
                        request_id <= candidate_id;
                        request_q_flat <= q_flat;
                        request_k_flat <= k_flat;
                        request_scale_product <= scale_product;
                        if (clear_before_update)
                            previous_score <= '0;
                        else
                            previous_score <= score_mem[candidate_id];
                        state <= COMPUTE;
                    end
                end
                COMPUTE: begin
                    pending_score <= dot_score_out;
                    pending_contribution_saturated <= dot_contribution_saturated;
                    pending_score_saturated <= dot_score_saturated;
                    state <= WRITE_RESULT;
                end
                WRITE_RESULT: begin
                    score_mem[request_id] <= pending_score;
                    result_valid <= 1'b1;
                    result_candidate_id <= request_id;
                    result_score <= pending_score;
                    contribution_saturated <= pending_contribution_saturated;
                    score_saturated <= pending_score_saturated;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
