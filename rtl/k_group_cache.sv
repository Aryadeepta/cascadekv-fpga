// Ordered INT4 K-group routing cache.
//
// Storage is one 80-bit word per (candidate, group): {scale, k_flat}.  A
// single logical word keeps the independently programmed K payload and scale
// together while still allowing generic synthesis tools to infer one RAM.
// K nibbles are opaque here: k_flat[i*4 +: 4] is lane i and is preserved
// bit-for-bit.
//
// For the default GROUP_COUNT=8 / GROUP_WIDTH=3 layout, flat_index is the
// concatenation {candidate_id, group_index}, i.e. candidate_id * 8 +
// group_index.  This deliberately avoids a general multiplier.
//
// A request handshakes on a rising edge with request_valid && request_ready.
// Its synchronous-read response is valid in the following cycle.  The
// response is a one-entry ready/valid buffer: while response_valid is high
// and response_ready is low, every response field remains stable and no new
// request is accepted.  An accepting response can be replaced on the same
// edge by a new request.
//
// reset clears only response/control state.  cache_mem is intentionally never
// reset, so it remains RAM-friendly and entries must be programmed before use.
module k_group_cache #(
    parameter integer CANDIDATE_COUNT = 256,
    parameter integer GROUP_COUNT = 8,
    parameter integer ID_WIDTH = $clog2(CANDIDATE_COUNT),
    parameter integer GROUP_WIDTH = 3
) (
    input  logic                         clk,
    input  logic                         rst,

    input  logic                         write_valid,
    input  logic [ID_WIDTH-1:0]          write_candidate_id,
    input  logic [GROUP_WIDTH-1:0]       write_group_index,
    input  logic [63:0]                  write_k_flat,
    input  logic signed [15:0]           write_scale,

    input  logic                         request_valid,
    output logic                         request_ready,
    input  logic [ID_WIDTH-1:0]          candidate_id,
    input  logic [GROUP_WIDTH-1:0]       group_index,

    output logic                         response_valid,
    input  logic                         response_ready,
    output logic [63:0]                  response_k_flat,
    output logic signed [15:0]           response_scale,
    output logic [ID_WIDTH-1:0]          response_candidate_id,
    output logic [GROUP_WIDTH-1:0]       response_group_index
);
    localparam integer CACHE_DEPTH = CANDIDATE_COUNT * GROUP_COUNT;

    // Default dimensions make each address exactly {candidate_id, group_index}.
    logic [79:0] cache_mem [0:CACHE_DEPTH-1];
    logic [ID_WIDTH+GROUP_WIDTH-1:0] write_flat_index;
    logic [ID_WIDTH+GROUP_WIDTH-1:0] request_flat_index;

    assign write_flat_index = {write_candidate_id, write_group_index};
    assign request_flat_index = {candidate_id, group_index};

    // One response slot provides backpressure without an additional request FIFO.
    assign request_ready = !response_valid || response_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            response_valid <= 1'b0;
            response_k_flat <= '0;
            response_scale <= '0;
            response_candidate_id <= '0;
            response_group_index <= '0;
        end else begin
            if (write_valid)
                cache_mem[write_flat_index] <= {write_scale, write_k_flat};

            if (request_valid && request_ready) begin
                {response_scale, response_k_flat} <= cache_mem[request_flat_index];
                response_candidate_id <= candidate_id;
                response_group_index <= group_index;
                response_valid <= 1'b1;
            end else if (response_valid && response_ready) begin
                response_valid <= 1'b0;
            end
        end
    end
endmodule
