// Offline Host-only report. Reading existing CSVs never starts an NPU or MPI job.
#include "profile.hpp"
#include "perfetto.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>

namespace {
using namespace CombineProfile;
namespace fs = std::filesystem;
std::vector<std::string> split(std::string line)
{
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> result;
    size_t start = 0;
    for (;;) {
        auto end = line.find(',', start);
        result.push_back(line.substr(start, end - start));
        if (end == std::string::npos) return result;
        start = end + 1;
    }
}
uint64_t integer(const std::string& s)
{
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid integer: " + s);
    return std::stoull(s);
}
struct Csv {
    std::ifstream file;
    std::vector<std::string> header, cells;
    explicit Csv(const fs::path& path) : file(path)
    {
        std::string line;
        if (!file || !std::getline(file, line)) throw std::runtime_error("Cannot read " + path.string());
        header = split(line);
    }
    bool next()
    {
        std::string line;
        if (!std::getline(file, line)) return false;
        cells = split(line);
        if (cells.size() != header.size()) throw std::runtime_error("Truncated CSV row");
        return true;
    }
    int column(const std::string& name) const
    {
        auto it = std::find(header.begin(), header.end(), name);
        return it == header.end() ? -1 : int(it - header.begin());
    }
    std::string get(const std::string& name) const
    {
        const int c = column(name);
        if (c < 0) throw std::runtime_error("Missing CSV column: " + name);
        return cells[c];
    }
};
std::vector<std::string> field_names()
{
    std::ostringstream out;
    llbench::profile_header(out);
    auto line = out.str();
    line.pop_back();
    auto names = split(line);
    names.erase(names.begin(), names.begin() + 4);
    return names;
}
const auto names = field_names();
struct Record {
    std::array<uint64_t, Count> v{};
    std::bitset<Count> present;
    uint32_t iteration = 0, device = 0;
    bool receive = false;
    uint64_t operator[](Word w) const
    {
        return v[w];
    }
    bool interval(Word begin, Word end, uint64_t& ticks) const
    {
        if (!present[begin] || !present[end]) return false;
        // CSV timestamps are already offsets, except entry_tick itself.
        const uint64_t start = begin == Entry ? 0 : v[begin];
        if (v[end] < start) throw std::runtime_error("Non-monotonic profile interval");
        ticks = v[end] - start;
        return true;
    }
};
struct Phase {
    const char* name;
    Word begin, end;
    Word detailBegin = Count, detailEnd = Count;
};
constexpr Phase phases[] = {{"body", Entry, FlushDone},
                            {"profile_setup", Entry, ProfileInitEnd},
                            {"magic", ProfileInitEnd, MagicEnd},
                            {"init", MagicEnd, InitEnd, InitPrefixTicks, InitPrefixTicks},
                            {"stage", StageBegin, StageEnd, StageSelectLoadTicks, StagePublishTicks},
                            {"notify_wait", NotifyBegin, NotifyLocalReady, NotifyPollLoadTicks, NotifyPollCheckTicks},
                            {"notify_publish", NotifyLocalReady, NotifyEnd, NotifyResetTicks, NotifyPublishTicks},
                            {"group_ready_window", ReadyBegin, ReadyObserved, ReadyPollLoadTicks, ReadyResetTicks},
                            {"ready_retire", ReadyRetireBegin, ReadyRetireEnd, ReadySnapshotTicks, ReadyConsumedWaitTicks},
                            {"partial", PartialBegin, PartialEnd, PlanLoadTicks, PartialInputDrainTicks},
                            {"plan", PlanBegin, PlanEnd, PlanLoadTicks, PlanSelectTicks},
                            {"direct_prepare", DirectBegin, DirectSetupEnd, DirectRouteLoadTicks, DirectSetupTicks},
                            {"direct_send", DirectSetupEnd, DirectEnd, DirectScanTicks, DirectDrainTicks},
                            {"final", FinalBegin, FinalEnd, FinalMetaLoadTicks, EarlyStalls},
                            {"final_drain", FinalEnd, DrainEnd},
                            {"send_pipe_tail", DirectEnd, PipeDone},
                            {"recv_pipe_tail", DrainEnd, PipeDone},
                            {"batch_flush", PipeDone, FlushDone}};
constexpr size_t phaseCount = sizeof(phases) / sizeof(phases[0]);
struct Winner {
    uint64_t ticks = 0;
    Record record;
};
struct Event {
    double us = 0;
    uint32_t rank = 0;
};
struct Sample {
    uint32_t iteration = 0;
    Event event;
    std::vector<Record> records;
};
struct Case {
    std::map<uint32_t, Event> events;
    std::map<uint32_t, std::set<uint32_t>> eventRanks;
    std::array<std::map<uint32_t, Winner>, phaseCount> maxima;
    std::map<std::pair<uint32_t, uint32_t>, std::set<uint32_t>> seen;
    Sample median, worst;
    std::vector<llbench::TraceRow> traceMedian, traceWorst;
    uint32_t world = 0, level = 0, schema = 0;
};
void identify(std::ostream& out, const Record& r)
{
    out << "iter=" << r.iteration << " rank=" << r[Rank] << " dev=" << r.device << " core=" << r[Core];
}
std::vector<Word> top_costs(const Record& r, Word begin, Word end)
{
    std::vector<Word> costs;
    if (r[Level] < 2) return costs;
    for (uint32_t w = begin; w <= end && w < Count; ++w) {
        const auto& name = names[w];
        if (name.size() >= 6 && name.compare(name.size() - 6, 6, "_ticks") == 0 &&
            name.find("_max_") == std::string::npos && w != RdmaDeferGapTicks && r.v[w])
            costs.push_back(Word(w));
    }
    std::stable_sort(costs.begin(), costs.end(), [&](Word a, Word b) { return r[a] > r[b]; });
    return costs;
}
void detail(std::ostream& out, const Record& r)
{
    out << "  " << (r.receive ? "RECEIVE" : "SEND") << " longest_body ";
    identify(out, r);
    out << " body_ticks=" << r[FlushDone];
    // The kernel records lane counts only for final aggregation, not send cores.
    if (r.receive) out << " final_lanes=" << r[InputLanes] << '/' << r[OutputLanes];
    out << '\n';
    out << "    work: stage=" << r[StageRows] << " direct=" << r[DirectRows] << " partial=" << r[PartialRows]
        << " final_tokens=" << r[FinalTokens];
    if (r[Level] >= 2)
        out << " contributions=" << r[Contributions] << " local/get=" << r[PartialLocalContrib] << '/'
            << r[PartialGetContrib];
    out << '\n' << "    offsets_ticks:";
    const std::vector<Word> send = {InitEnd,          StageEnd,  NotifyLocalReady, NotifyEnd, DirectFirstSubmit,
                                    DirectLastSubmit, DirectEnd, PipeDone,         FlushDone};
    const std::vector<Word> receive = {ReadyEnd,       PartialEnd, FinalFirstReady, FinalLastReady,
                                       FinalLastStore, DrainEnd,   PipeDone,        FlushDone};
    for (Word w : r.receive ? receive : send)
        if (r.present[w]) out << ' ' << names[w] << '=' << r[w];
    out << '\n';
    const Word tailBegin = r.receive ? DrainEnd : DirectEnd;
    uint64_t tail = 0;
    if (r.interval(tailBegin, PipeDone, tail)) out << "    pipe_tail_ticks=" << tail << '\n';
    if (r[Level] < 2) return;
    if (r[RdmaPackets] && r[Schema] >= 3U) {
        out << "    rdma_return: packets=" << r[RdmaPackets] << " deferred=" << r[RdmaDeferredPackets];
        for (Word w : {RdmaFirstStage, RdmaLastStage, RdmaFirstPut, RdmaLastPut})
            if (r.present[w]) out << ' ' << names[w] << '=' << r[w];
        out << " stage_wait_ticks=" << r[RdmaStageWaitTicks]
            << " max=" << r[RdmaStageWaitMaxTicks] << "(peer=" << r[RdmaStageWaitMaxPeer] << ')'
            << " defer_gap_ticks=" << r[RdmaDeferGapTicks] << '\n';
    }
    if (r.receive && r[BatchGroupCalls] + r[BatchWholeTokens])
        out << "    batch: group_checks=" << r[BatchGroupCalls] << " whole_tokens=" << r[BatchWholeTokens]
            << " route_ticks=" << r[BatchRouteTicks] << '\n';
    if (r.receive && r[ReadySnapshotTicks])
        out << "    ready: first/all/done=" << r[ReadyFirstTask] << '/' << r[ReadyObserved] << '/' << r[ReadyTasksDone]
            << " before_all_observed/tasks/remote=" << r[ReadyTasksBeforeAll] << '/' << r[ReadyRemoteBeforeAll]
            << " setup/plan/filter/snapshot=" << r[ReadySetupTicks] << '/' << r[ReadyPlanTicks] << '/' << r[ReadyFilterTicks] << '/' << r[ReadySnapshotTicks]
            << " filter_skips/cached_tasks/metadata_reloads/pipeline_issue_ticks=" << r[ReadyFilterSkips] << '/'
            << r[ReadyCachedTasks] << '/' << r[ReadyMetadataReloads] << '/' << r[ReadyPipelineIssueTicks]
            << " ack_wait/consumed_wait=" << r[ReadyAckWaitTicks] << '/' << r[ReadyConsumedWaitTicks]
            << (r[ReadyTaskRounds] ? " (group_ready_window includes useful work)\n" : "\n");
    if (r.receive && (r[RouteLoadBatches] || r[RouteSnapshotCapacity]))
        out << "    route: batches/rows/bytes/group_passes=" << r[RouteLoadBatches] << '/' << r[RouteLoadedRows]
            << '/' << r[RouteLoadBytes] << '/' << r[RouteSelectPasses]
            << " snapshot_bytes/capacity/fallbacks=" << r[RouteSnapshotBytes] << '/' << r[RouteSnapshotCapacity]
            << '/' << r[RouteSnapshotFallbacks] << '\n';
    if (r.receive && r[LocalContributions] + r[RemoteContributions]) {
        out << "    final_local: contributions=" << r[LocalContributions]
            << " probes/misses=" << r[LocalProbeCalls] << '/' << r[LocalProbeMisses]
            << " probe_ticks=" << r[LocalProbeTicks] << " reduce_ticks=" << r[LocalReduceTicks] << '\n';
        out << "    final_remote: contributions=" << r[RemoteContributions]
            << " probes/misses=" << r[RemoteProbeCalls] << '/' << r[RemoteProbeMisses]
            << " probe_ticks=" << r[RemoteProbeTicks] << " reduce_ticks=" << r[RemoteReduceTicks] << '\n';
    }
    const auto costs = top_costs(r, kDetailBegin, kDetailEnd);
    for (size_t i = 0; i < std::min<size_t>(8, costs.size()); ++i) {
        if (i % 4 == 0) out << "    top_ticks:";
        out << ' ' << names[costs[i]] << '=' << r[costs[i]];
        if (i % 4 == 3 || i + 1 == std::min<size_t>(8, costs.size())) out << '\n';
    }
    out << "    polls calls/misses: local=" << r[NotifyPollCalls] << '/' << r[NotifyPollMisses]
        << " group=" << r[ReadyPollCalls] << '/' << r[ReadyPollMisses] << " data=" << r[ProbeCalls] << '/'
        << r[ProbeMisses] << " miss_ticks=" << r[ProbeMissTicks] << " early_stalls=" << r[EarlyStalls] << '\n';
    out << "    submit calls/remote: direct=" << r[DirectSubmitCalls] << '/' << r[DirectRemoteCalls]
        << " partial=" << r[PartialSubmitCalls] << '/' << r[PartialRemoteCalls]
        << " rdma_packets/quiet=" << r[RdmaPackets] << '/' << r[RdmaQuietCalls] << '\n';
    out << "    max_single_ticks:";
    if (r[DirectSubmitCalls])
        out << " direct=" << r[DirectSubmitMaxTicks] << "(peer=" << r[DirectSubmitMaxPeer]
            << ",token=" << r[DirectSubmitMaxToken] << ')';
    if (r[PartialSubmitCalls])
        out << " partial=" << r[PartialSubmitMaxTicks] << "(peer=" << r[PartialSubmitMaxPeer]
            << ",token=" << r[PartialSubmitMaxToken] << ')';
    if (r[ProbeCalls]) {
        out << " probe=" << r[ProbeMaxTicks] << "(token=" << r[ProbeMaxToken] << ",slot=";
        if (r[ProbeMaxSlot] == 0xffffffffU)
            out << "all";
        else
            out << r[ProbeMaxSlot];
        out << ')';
    }
    if (r[RdmaQuietCalls]) out << " quiet=" << r[RdmaQuietMaxTicks] << "(peer=" << r[RdmaQuietMaxPeer] << ')';
    out << '\n';
}
void sample(std::ostream& out, const char* label, const Sample& s)
{
    out << label << " iter=" << s.iteration << " scope_max_Event_us=" << s.event.us << " Event_rank=" << s.event.rank
        << '\n';
    for (bool receive : {false, true}) {
        const Record* longest = nullptr;
        std::vector<uint64_t> spans;
        for (const auto& r : s.records)
            if (r.receive == receive) {
                spans.push_back(r[FlushDone]);
                if (!longest || r[FlushDone] > (*longest)[FlushDone]) longest = &r;
            }
        if (!longest) continue;
        std::sort(spans.begin(), spans.end());
        out << "  " << (receive ? "receive" : "send") << " cores=" << spans.size()
            << " body_ticks min/median/max=" << spans.front() << '/' << spans[(spans.size() - 1) / 2] << '/'
            << spans.back() << '\n';
        detail(out, *longest);
    }
    // Also expose a producer even when the slowest body belongs to a consumer.
    // This ranks durations from each core's own entry, not absolute launch time.
    const Record* latest = nullptr;
    for (const auto& r : s.records)
        if (r.present[RdmaLastPut] && (!latest || r[RdmaLastPut] > (*latest)[RdmaLastPut])) latest = &r;
    if (latest) {
        out << "  RETURN largest_last_put_offset (own-core entry) ";
        identify(out, *latest);
        out << " first=" << (*latest)[RdmaFirstPut] << " last=" << (*latest)[RdmaLastPut]
            << " packets=" << (*latest)[RdmaPackets] << " deferred=" << (*latest)[RdmaDeferredPackets]
            << " stage_wait=" << (*latest)[RdmaStageWaitTicks]
            << " defer_gap=" << (*latest)[RdmaDeferGapTicks] << '\n';
    }
}
void report(const fs::path& directory, uint64_t traceFrequency = 0)
{
    std::map<uint32_t, fs::path> files;
    const std::regex pattern("rank-([0-9]+)-profile\\.csv");
    for (const auto& f : fs::directory_iterator(directory)) {
        std::smatch match;
        const auto name = f.path().filename().string();
        if (f.is_regular_file() && std::regex_match(name, match, pattern))
            files.emplace(uint32_t(integer(match[1].str())), f.path());
    }
    if (files.empty()) throw std::runtime_error("No rank-N-profile.csv in " + directory.string());
    std::map<uint32_t, Case> cases;
    // Event files are small. Choose real, matching samples before streaming the large profile CSVs.
    for (const auto& [rank, path] : files) {
        Csv csv(directory / ("rank-" + std::to_string(rank) + ".csv"));
        while (csv.next()) {
            const auto bs = uint32_t(integer(csv.get("tokens"))), iter = uint32_t(integer(csv.get("iteration")));
            const double us = std::stod(csv.get("combine_event_us"));
            if (!std::isfinite(us) || us <= 0) throw std::runtime_error("Invalid Event time");
            auto& c = cases[bs];
            if (!c.eventRanks[iter].insert(rank).second) throw std::runtime_error("Duplicate Event sample");
            auto& best = c.events[iter];
            if (us > best.us) best = {us, rank};
        }
    }
    for (auto& [bs, c] : cases) {
        std::vector<std::pair<uint32_t, Event>> ordered(c.events.begin(), c.events.end());
        for (const auto& [iter, ranks] : c.eventRanks)
            if (ranks.size() != files.size()) throw std::runtime_error("Missing Event rank/sample");
        std::stable_sort(ordered.begin(), ordered.end(),
                         [](const auto& a, const auto& b) { return a.second.us < b.second.us; });
        const auto& mid = ordered[(ordered.size() - 1) / 2];
        c.median = {mid.first, mid.second, {}};
        c.worst = {ordered.back().first, ordered.back().second, {}};
    }
    for (const auto& [rank, path] : files) {
        Csv csv(path);
        std::array<int, Count> columns;
        for (uint32_t w = 0; w < Count; ++w) columns[w] = csv.column(names[w]);
        while (csv.next()) {
            Record r;
            r.iteration = uint32_t(integer(csv.get("iteration")));
            r.device = uint32_t(integer(csv.get("device")));
            const auto role = csv.get("role");
            if (role != "send" && role != "receive") throw std::runtime_error("Invalid core role");
            r.receive = role == "receive";
            for (uint32_t w = 0; w < Count; ++w)
                if (columns[w] >= 0 && !csv.cells[columns[w]].empty()) {
                    r.v[w] = integer(csv.cells[columns[w]]);
                    r.present[w] = true;
                }
            if (r[Schema] < 2U || r[Schema] > kSchema || r[Complete] != 1 ||
                r[Rank] != rank || !r.present[FlushDone] ||
                !r.present[Entry] || !r[World] || rank >= r[World] || (r[Level] != 1 && r[Level] != 2))
                throw std::runtime_error("Incomplete profile or unsupported schema (requires schema 2..8)");
            auto it = cases.find(uint32_t(r[Tokens]));
            if (it == cases.end() || !it->second.events.count(r.iteration))
                throw std::runtime_error("Profile has no matching Event sample");
            auto& c = it->second;
            if (c.world && (c.world != r[World] || c.level != r[Level] || c.schema != r[Schema]))
                throw std::runtime_error("Mixed profile configuration");
            c.world = uint32_t(r[World]);
            c.level = uint32_t(r[Level]);
            c.schema = uint32_t(r[Schema]);
            if (!c.seen[{r.iteration, rank}].insert(uint32_t(r[Core])).second)
                throw std::runtime_error("Duplicate profile core");
            for (size_t p = 0; p < phaseCount; ++p) {
                uint64_t ticks = 0;
                if (!r.interval(phases[p].begin, phases[p].end, ticks)) continue;
                auto& byIter = c.maxima[p];
                auto winner = byIter.find(r.iteration);
                if (winner == byIter.end() || ticks > winner->second.ticks) byIter[r.iteration] = {ticks, r};
            }
            for (auto* s : {&c.median, &c.worst})
                if (r.iteration == s->iteration && rank == s->event.rank) s->records.push_back(r);
            if (traceFrequency) {
                const llbench::TraceRow trace{r.v, r.present, r.device, r.receive};
                if (r.iteration == c.median.iteration) c.traceMedian.push_back(trace);
                if (r.iteration == c.worst.iteration) c.traceWorst.push_back(trace);
            }
        }
    }
    // Validate coverage before publishing any trace.
    for (const auto& [bs, c] : cases) {
        if (!c.world) throw std::runtime_error("Missing profile records");
        const auto& cores = c.seen.begin()->second;
        if (cores.empty() || *cores.begin() != 0 || *cores.rbegin() + 1 != cores.size())
            throw std::runtime_error("Non-contiguous profile core set");
        for (const auto& [iter, event] : c.events)
            for (const auto& [rank, path] : files) {
                auto it = c.seen.find({iter, rank});
                if (it == c.seen.end() || it->second != cores)
                    throw std::runtime_error("Missing profile rank/core/sample");
            }
    }
    if (traceFrequency) {
        // Fresh export directory also permits repeated conversion with another frequency.
        auto destination = directory / "perfetto-local";
        for (uint32_t suffix = 1; !fs::create_directory(destination); ++suffix)
            destination = directory / ("perfetto-local-" + std::to_string(suffix));
        for (const auto& [bs, c] : cases) {
            for (bool worst : {false, true}) {
                if (worst && c.median.iteration == c.worst.iteration) continue;
                const auto& s = worst ? c.worst : c.median;
                const std::string label = worst ? "worst" : (c.median.iteration == c.worst.iteration ? "p50-worst" : "p50");
                const auto file = destination / ("bs" + std::to_string(bs) + "-" + label + "-i" +
                                                  std::to_string(s.iteration) + ".perfetto.json");
                llbench::write_perfetto(file, worst ? c.traceWorst : c.traceMedian,
                    {bs, s.iteration, c.world, traceFrequency, s.event.us,
                     "available rank-MAX " + label + "; coverage=" + std::to_string(files.size()) + "/" + std::to_string(c.world),
                     directory.string()});
            }
        }
        std::cout << "Perfetto: " << destination.string() << " (local CSV coverage; independent core origins)\n";
    }
    std::cout
        << "=== LL_COMBINE_PROFILE_SUMMARY BEGIN ===\n"
        << "Event unit=us; device durations/offsets=ticks. No assumed clock frequency.\n"
        << "Phase rows: mean/max of per-iteration maximum over available cores; overlapping rows are not additive.\n"
        << "Detail: same Event sample/rank, longest-duration send/receive core; top counters may overlap.\n"
        << "Scalar-clock scopes: *_V/*_MTE waits do not synchronize Scalar; later V_S/PIPE_ALL may include prior "
           "work.\n"
        << "Profiling adds overhead. Core offsets use each core's own entry; no cross-rank clock subtraction.\n";
    // Preserve shape/build context when node 0's summary exists.
    if (fs::exists(directory / "summary.csv")) {
        Csv csv(directory / "summary.csv");
        while (csv.next()) {
            std::cout << "CONFIG";
            for (const char* key : {"tokens", "hidden", "topk", "experts", "dtype", "route", "seed", "warmup",
                                    "small_batch", "early_reduce", "local_first", "batch_final", "recv_cores", "remote_first", "defer_return",
                                    "entry_sync", "source_id"})
                if (csv.column(key) >= 0) std::cout << ' ' << key << '=' << csv.get(key);
            std::cout << '\n';
        }
    }
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& [bs, c] : cases) {
        std::cout << "\nBS=" << bs << " EP=" << c.world << " level=" << c.level << " scope_ranks=";
        bool first = true;
        for (const auto& [rank, path] : files) {
            std::cout << (first ? "" : ",") << rank;
            first = false;
        }
        std::cout << " coverage=" << files.size() << '/' << c.world
                  << (files.size() == c.world ? " FULL" : " PARTIAL: local files only")
                  << " samples=" << c.events.size() << '\n';
        double sum = 0, minimum = c.worst.event.us;
        for (const auto& [iter, e] : c.events) {
            sum += e.us;
            minimum = std::min(minimum, e.us);
        }
        std::cout << "scope rank-MAX Event us mean/min/max=" << sum / c.events.size() << '/' << minimum << '/'
                  << c.worst.event.us << '\n';
        std::cout
            << "phase                  mean_ticks    max_ticks  worst(iter/rank/core)  top3 counters of that record\n";
        for (size_t p = 0; p < phaseCount; ++p) {
            const auto& values = c.maxima[p];
            if (values.empty()) continue;
            long double total = 0;
            const Winner* worst = nullptr;
            for (const auto& [iter, w] : values) {
                total += w.ticks;
                if (!worst || w.ticks > worst->ticks) worst = &w;
            }
            std::cout << std::left << std::setw(21) << phases[p].name << std::right << std::setw(12)
                      << double(total / values.size()) << std::setw(13) << worst->ticks << "  "
                      << worst->record.iteration << '/' << worst->record[Rank] << '/' << worst->record[Core];
            if (values.size() != c.events.size()) std::cout << " n=" << values.size();
            const auto costs = top_costs(worst->record, phases[p].detailBegin, phases[p].detailEnd);
            for (size_t i = 0; i < std::min<size_t>(3, costs.size()); ++i)
                std::cout << ' ' << names[costs[i]] << '=' << worst->record[costs[i]];
            std::cout << '\n';
        }
        sample(std::cout, "MEDIAN_EVENT_SAMPLE", c.median);
        if (c.worst.iteration != c.median.iteration)
            sample(std::cout, "WORST_EVENT_SAMPLE", c.worst);
        else
            std::cout << "WORST_EVENT_SAMPLE is the same sample.\n";
    }
    std::cout << "=== LL_COMBINE_PROFILE_SUMMARY END ===\n";
}
#include "compact_report.hpp"
}  // namespace
int main(int argc, char** argv)
{
    const bool suite = argc == 3 && std::string(argv[1]) == "--suite";
    const bool trace = (argc == 3 || argc == 4) && std::string(argv[1]) == "--perfetto";
    if ((!suite && !trace && argc != 2) || (argc >= 2 && std::string(argv[1]) == "--help")) {
        std::cout << "Usage: ll_combine_profile_report RESULT_PHASE_DIRECTORY | --suite RUN_DIRECTORY | --perfetto PHASE_DIRECTORY [FREQUENCY_HZ]\n";
        return argc == 2 ? 0 : 2;
    }
    try {
        if (suite) suite_report(argv[2]);
        else if (trace) {
            const auto frequency = argc == 4 ? integer(argv[3]) : 1000000000ULL;
            if (!frequency) throw std::runtime_error("Perfetto frequency must be positive");
            report(argv[2], frequency);
        }
        else report(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Profile summary failed: " << e.what() << '\n';
        return 1;
    }
}
