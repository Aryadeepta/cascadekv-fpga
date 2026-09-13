// Ordered work scheduler for the progressive 16/32/64/128-D score stages.
//
// Stage encodings are 0=16D, 1=32D, 2=64D, and 3=128D.  A start presented
// while busy is ignored.  Survivor writes are independent of scheduling and
// are intentionally not cleared by reset so the array remains RAM-friendly.
module cascade_stage_controller #(
    parameter integer CANDIDATE_CAPACITY = 256,
    parameter integer ID_WIDTH = $clog2(CANDIDATE_CAPACITY),
    parameter integer INDEX_WIDTH = $clog2(CANDIDATE_CAPACITY),
    parameter integer COUNT_WIDTH = $clog2(CANDIDATE_CAPACITY + 1)
) (
    input  logic                   clk,
    input  logic                   rst,

    input  logic                   start,
    input  logic [1:0]             stage,
    input  logic [COUNT_WIDTH-1:0] stage_candidate_count,

    input  logic                   survivor_write_valid,
    input  logic [INDEX_WIDTH-1:0] survivor_write_index,
    input  logic [ID_WIDTH-1:0]    survivor_write_candidate_id,

    output logic                   work_valid,
    input  logic                   work_ready,
    output logic [ID_WIDTH-1:0]    work_candidate_id,
    output logic [2:0]             work_group_index,
    output logic                   work_clear_before_update,
    output logic                   busy,
    output logic                   done
);
    // This is written synchronously and read for the next work item.  It is
    // not reset, both to preserve programmed entries and enable memory
    // inference in generic synthesis.
    logic [ID_WIDTH-1:0] survivor_mem [0:CANDIDATE_CAPACITY-1];

    logic [1:0] active_stage;
    logic [COUNT_WIDTH-1:0] active_count;
    logic [COUNT_WIDTH-1:0] candidate_index;
    logic [2:0] group_offset;

    function automatic logic [2:0] first_group(input logic [1:0] selected_stage);
        case (selected_stage)
            2'd0: first_group = 3'd0;
            2'd1: first_group = 3'd1;
            2'd2: first_group = 3'd2;
            default: first_group = 3'd4;
        endcase
    endfunction

    function automatic logic [2:0] group_count(input logic [1:0] selected_stage);
        case (selected_stage)
            2'd0, 2'd1: group_count = 3'd1;
            2'd2: group_count = 3'd2;
            default: group_count = 3'd4;
        endcase
    endfunction

    function automatic logic [ID_WIDTH-1:0] candidate_for(
        input logic [1:0] selected_stage,
        input logic [INDEX_WIDTH-1:0] selected_index
    );
        if (selected_stage == 2'd0)
            candidate_for = selected_index;
        else
            candidate_for = survivor_mem[selected_index];
    endfunction

    always_ff @(posedge clk) begin
        if (survivor_write_valid)
            survivor_mem[survivor_write_index] <= survivor_write_candidate_id;

        if (rst) begin
            work_valid <= 1'b0;
            work_candidate_id <= '0;
            work_group_index <= '0;
            work_clear_before_update <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            active_stage <= '0;
            active_count <= '0;
            candidate_index <= '0;
            group_offset <= '0;
        end else begin
            done <= 1'b0;
            if (!busy) begin
                // Starts are sampled only while idle.  A zero-length stage
                // completes immediately, with done high for this clock.
                if (start) begin
                    active_stage <= stage;
                    active_count <= stage_candidate_count;
                    candidate_index <= '0;
                    group_offset <= '0;
                    if (stage_candidate_count == '0) begin
                        work_valid <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        busy <= 1'b1;
                        work_valid <= 1'b1;
                        work_candidate_id <= candidate_for(stage, '0);
                        work_group_index <= first_group(stage);
                        work_clear_before_update <= (stage == 2'd0);
                    end
                end
            end else if (work_valid && work_ready) begin
                if (group_offset + 3'd1 < group_count(active_stage)) begin
                    group_offset <= group_offset + 3'd1;
                    work_group_index <= first_group(active_stage) + group_offset + 3'd1;
                end else if (candidate_index + 1'b1 < active_count) begin
                    candidate_index <= candidate_index + 1'b1;
                    group_offset <= '0;
                    work_candidate_id <= candidate_for(
                        active_stage, candidate_index[INDEX_WIDTH-1:0] + 1'b1
                    );
                    work_group_index <= first_group(active_stage);
                end else begin
                    work_valid <= 1'b0;
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
