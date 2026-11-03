#pragma once
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <vector>
#include "../../csrc/kernels/combine/low_latency/combine_profile_layout.h"

namespace llbench {
inline void profile_header(std::ostream& out)
{
    out << "iteration,device,role,profile_build,";
    // Timestamps except entry_tick are offsets from this core's own entry.
    constexpr const char* names[] = {
        "entry_tick",
        "profile_init_end",
        "magic_end",
        "init_count_end",
        "init_layout_end",
        "init_end",
        "stage_begin",
        "stage_selected",
        "stage_copies_end",
        "stage_drained",
        "stage_end",
        "notify_begin",
        "notify_local_ready",
        "notify_end",
        "ready_begin",
        "ready_end",
        "partial_begin",
        "partial_init_end",
        "plan_begin",
        "plan_end",
        "partial_run_begin",
        "partial_run_end",
        "partial_end",
        "final_begin",
        "final_init_end",
        "final_first_ready",
        "final_last_ready",
        "final_last_store",
        "final_end",
        "drain_end",
        "direct_begin",
        "direct_types_end",
        "direct_routes_end",
        "direct_prepared",
        "direct_setup_end",
        "direct_first_submit",
        "direct_last_submit",
        "direct_pending_end",
        "direct_end",
        "pipe_done",
        "flush_done",
        "rdma_first_stage",
        "rdma_last_stage",
        "rdma_first_put",
        "rdma_last_put",
        "stage_rows",
        "direct_rows",
        "partial_rows",
        "final_tokens",
        "input_lanes",
        "output_lanes",
        "init_prefix_ticks",
        "stage_select_load_ticks",
        "stage_select_vector_ticks",
        "stage_metadata_ticks",
        "stage_reuse_wait_ticks",
        "stage_input_ticks",
        "stage_store_issue_ticks",
        "stage_drain_ticks",
        "stage_publish_ticks",
        "notify_poll_load_ticks",
        "notify_poll_check_ticks",
        "notify_poll_calls",
        "notify_poll_misses",
        "notify_reset_ticks",
        "notify_publish_ticks",
        "ready_poll_load_ticks",
        "ready_poll_check_ticks",
        "ready_poll_calls",
        "ready_poll_misses",
        "ready_reset_ticks",
        "direct_route_load_ticks",
        "direct_select_vector_ticks",
        "direct_gather_ticks",
        "direct_setup_ticks",
        "direct_scan_ticks",
        "direct_metadata_ticks",
        "direct_reuse_wait_ticks",
        "direct_input_issue_ticks",
        "direct_input_wait_ticks",
        "direct_output_wait_ticks",
        "direct_pack_ticks",
        "direct_submit_ticks",
        "direct_submit_calls",
        "direct_remote_calls",
        "direct_submit_max_ticks",
        "direct_submit_max_peer",
        "direct_submit_max_token",
        "direct_drain_ticks",
        "plan_load_ticks",
        "plan_select_ticks",
        "partial_scan_ticks",
        "partial_scan_load_ticks",
        "partial_scan_vector_ticks",
        "partial_metadata_ticks",
        "partial_reuse_wait_ticks",
        "partial_local_issue_ticks",
        "partial_get_issue_ticks",
        "partial_local_wait_ticks",
        "partial_get_wait_ticks",
        "partial_compute_issue_ticks",
        "partial_output_wait_ticks",
        "partial_pack_ticks",
        "partial_submit_ticks",
        "partial_submit_calls",
        "partial_remote_calls",
        "partial_submit_max_ticks",
        "partial_submit_max_peer",
        "partial_submit_max_token",
        "partial_input_drain_ticks",
        "partial_local_contrib",
        "partial_get_contrib",
        "final_meta_load_ticks",
        "final_pending_build_ticks",
        "probe_ticks",
        "probe_load_ticks",
        "probe_check_ticks",
        "probe_calls",
        "probe_misses",
        "probe_flag_blocks",
        "probe_miss_ticks",
        "probe_max_ticks",
        "probe_max_token",
        "probe_max_slot",
        "final_input_issue_ticks",
        "final_input_wait_ticks",
        "final_reuse_wait_ticks",
        "final_compute_issue_ticks",
        "final_output_wait_ticks",
        "final_output_pack_ticks",
        "final_store_issue_ticks",
        "final_input_drain_ticks",
        "clear_ticks",
        "contributions",
        "early_advance_calls",
        "early_stalls",
        "rdma_stage_issue_ticks",
        "rdma_stage_wait_ticks",
        "rdma_put_ticks",
        "rdma_quiet_ticks",
        "rdma_packets",
        "rdma_quiet_calls",
        "rdma_quiet_max_ticks",
        "rdma_quiet_max_peer",
        "rdma_deferred_packets",
        "rdma_defer_gap_ticks",
        "rdma_stage_wait_max_ticks",
        "rdma_stage_wait_max_peer",
        "early_issue_ticks",
        "probe_confirmed_slots",
        "probe_blocked_token",
        "probe_blocked_slot",
        "metadata_batches",
        "tail_inspect_ticks",
        "tail_token",
        "tail_missing_mask",
        "tail_slot",
        "tail_block",
        "tail_missing_blocks",
        "tail_first_seen",
        "tail_last_seen",
        "tail_ready",
        "tail_observations",
        "poll_pause_ticks",
        "poll_pause_calls",
        "local_probe_ticks",
        "local_probe_calls",
        "local_probe_misses",
        "local_reduce_ticks",
        "local_contributions",
        "remote_probe_ticks",
        "remote_probe_calls",
        "remote_probe_misses",
        "remote_reduce_ticks",
        "remote_contributions",
        "batch_group_calls",
        "batch_whole_tokens",
        "batch_route_ticks",
        "rank",
        "core",
        "tokens",
        "world",
        "schema",
        "complete",
        "valid",
        "level",
    };
    static_assert(sizeof(names) / sizeof(names[0]) == CombineProfile::Count, "profile CSV fields");
    for (uint32_t w = 0; w < CombineProfile::Count; ++w) out << (w ? "," : "") << names[w];
    out << '\n';
}

inline uint64_t profile_interval(const uint64_t* row, CombineProfile::Word begin, CombineProfile::Word end)
{
    const uint64_t mask = (uint64_t(1) << begin) | (uint64_t(1) << end);
    if ((row[CombineProfile::Valid] & mask) != mask) return 0;
    if (row[end] < row[begin]) throw std::runtime_error("Non-monotonic Combine profile clock");
    return row[end] - row[begin];
}

inline void profile_rows(std::ostream& out, const std::vector<uint64_t>& data, uint32_t cores, uint32_t recv,
                         uint32_t rank, uint32_t world, uint32_t bs, uint32_t device, uint32_t level,
                         uint32_t iteration, std::vector<uint32_t> selection = {})
{
    using namespace CombineProfile;
    if (data.size() != size_t(cores) * kWords || recv > cores)
        throw std::runtime_error("Invalid Combine profile dimensions");
    if (selection.empty())
        for (uint32_t core = 0; core < cores; ++core) selection.push_back(core);
    for (uint32_t core : selection) {
        if (core >= cores) throw std::runtime_error("Invalid selected profile core");
        auto row = data.data() + core * kWords;
        const uint64_t required = (uint64_t(1) << Entry) | (uint64_t(1) << FlushDone);
        if (row[Schema] != kSchema || row[Complete] != 1 || row[Rank] != rank || row[Core] != core ||
            row[Tokens] != bs || row[World] != world || row[Level] != level || (row[Valid] & required) != required)
            throw std::runtime_error("Incomplete or mismatched Combine profile record");
        out << iteration << ',' << device << ',' << (core < cores - recv ? "send" : "receive") << ",1,";
        for (uint32_t w = 0; w < Count; ++w) {
            if (w) out << ',';
            if (w < StageRows) {
                if (row[Valid] & (uint64_t(1) << w))
                    out << (w == Entry ? row[w] : profile_interval(row, Entry, Word(w)));
            } else if (level >= 2 || w < kDetailBegin || w > kDetailEnd) {
                out << row[w];
            }
        }
        out << '\n';
    }
}
}  // namespace llbench
