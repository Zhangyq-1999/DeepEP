#pragma once
// Host-only Chrome Trace Event JSON export. No extra device timestamps.
#include "profile.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>

namespace llbench {
struct TraceRow {
    std::array<uint64_t, CombineProfile::Count> values{};
    std::bitset<CombineProfile::Count> present;
    uint32_t device = 0;
    bool receive = false;
};
inline TraceRow trace_row(const uint64_t* raw, uint32_t device, bool receive)
{
    using namespace CombineProfile;
    TraceRow r;
    r.device = device;
    r.receive = receive;
    for (uint32_t w = 0; w < Count; ++w) {
        r.present[w] = w < StageRows ? bool(raw[Valid] & (uint64_t(1) << w)) :
            (raw[Level] >= 2 || !IsDetailWord(w));
        if (r.present[w]) r.values[w] = w && w < StageRows ? profile_interval(raw, Entry, Word(w)) : raw[w];
    }
    return r;
}
inline std::string trace_quote(const std::string& s)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
inline std::vector<std::string> trace_fields()
{
    std::ostringstream header;
    profile_header(header);
    std::istringstream input(header.str());
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(field);
    fields.erase(fields.begin(), fields.begin() + 4);
    fields.back().pop_back(); // newline
    return fields;
}
struct TraceInfo {
    uint32_t tokens = 0, iteration = 0, world = 0;
    uint64_t frequencyHz = 1000000000;
    double eventUs = 0;
    std::string selection, source;
};
inline void write_perfetto(const std::filesystem::path& path, const std::vector<TraceRow>& rows,
                           const TraceInfo& info)
{
    using namespace CombineProfile;
    if (rows.empty() || !info.frequencyHz || !std::isfinite(info.eventUs) || info.eventUs <= 0)
        throw std::runtime_error("Invalid Perfetto sample or counter frequency");
    const auto fields = trace_fields();
    // Independent core origins deliberately preserve the existing CSV contract.
    // A duration is never inferred from accumulated counters or another core.
    constexpr const char* note = "Each core starts at its own entry (zero); cores/ranks are NOT clock-aligned. "
        "Profiled sample, not clean Event performance. Put/ready markers are local observations, not network latency. "
        "Cumulative counters are args only and may overlap. Event time is metadata, not an aligned slice.";
    struct Span { const char* name; Word begin, end; };
    constexpr Span spans[] = {
        {"profile setup", Entry, ProfileInitEnd}, {"magic", ProfileInitEnd, MagicEnd},
        {"initialization", MagicEnd, InitEnd},
        {"TERMINAL stage", StageBegin, StageEnd},
        {"stage select", StageBegin, StageSelected}, {"stage copies", StageSelected, StageCopiesEnd},
        {"stage drain", StageCopiesEnd, StageDrained}, {"stage publish", StageDrained, StageEnd},
        {"local ready wait", NotifyBegin, NotifyLocalReady}, {"ready publish", NotifyLocalReady, NotifyEnd},
        {"ready retirement", ReadyRetireBegin, ReadyRetireEnd},
        {"dependency-driven partial work", ReadyFirstTask, ReadyTasksDone},
        {"per-core SIMT publish", CoreNotifyBegin, CoreNotifyEnd},
        {"group ready polling", ReadyBegin, ReadyObserved},
        {"ready inspect and reset", ReadyObserved, ReadyEnd},
        {"group ready wait", ReadyBegin, ReadyEnd},
        {"FIRST_HOP partial", PartialBegin, PartialEnd},
        {"partial init", PartialBegin, PartialInitEnd}, {"partial plan", PlanBegin, PlanEnd},
        {"partial process", PartialRunBegin, PartialRunEnd}, {"partial drain", PartialRunEnd, PartialEnd},
        {"DIRECT prepare", DirectBegin, DirectSetupEnd},
        {"DIRECT send", DirectSetupEnd, DirectEnd},
        {"DIRECT pending", DirectSetupEnd, DirectPendingEnd}, {"DIRECT drain", DirectPendingEnd, DirectEnd},
        {"final aggregate", FinalBegin, FinalEnd}, {"final init", FinalBegin, FinalInitEnd},
        {"final drain", FinalEnd, DrainEnd}, {"flush", PipeDone, FlushDone},
        {"coalesce pack and send", CoalesceSendBegin, CoalesceSendEnd},
        {"coalesce receive and unpack", CoalesceSendEnd, CoalesceReceiveEnd},
    };
    struct Slice { std::string name; uint64_t begin, end; unsigned lane = 0; };
    std::set<std::pair<uint64_t, uint64_t>> seen;
    std::set<uint64_t> ranks;
    const auto temporary = path.string() + ".tmp";
    std::filesystem::create_directories(path.parent_path());
    try {
        std::ofstream out(temporary);
        if (!out) throw std::runtime_error("Cannot open Perfetto output");
        out << std::setprecision(17) << "{\"displayTimeUnit\":\"us\",\"otherData\":{\"note\":" << trace_quote(note)
            << ",\"clock_scope\":\"core-local\",\"frequency_hz\":" << info.frequencyHz
            << ",\"tokens\":" << info.tokens << ",\"world\":" << info.world << ",\"iteration\":" << info.iteration
            << ",\"selection\":" << trace_quote(info.selection) << ",\"sample_event_us\":" << info.eventUs
            << ",\"source\":" << trace_quote(info.source) << "},\"traceEvents\":[";
        bool comma = false;
        auto item = [&] { if (comma) out << ','; comma = true; out << '\n'; };
        auto us = [&](uint64_t tick) { return double(tick) * (1e6 / double(info.frequencyHz)); };
        auto metadata = [&](const char* type, uint64_t pid, uint64_t tid, const std::string& name) {
            item(); out << "{\"ph\":\"M\",\"name\":" << trace_quote(type) << ",\"pid\":" << pid
                        << ",\"tid\":" << tid << ",\"args\":{\"name\":" << trace_quote(name) << "}}";
        };
        for (const auto& r : rows) {
            const auto& v = r.values;
            if (!r.present[Entry] || !r.present[FlushDone] || v[Schema] < 2 || v[Schema] > kSchema ||
                v[Complete] != 1 || v[Tokens] != info.tokens || v[World] != info.world || v[Rank] >= info.world ||
                !seen.emplace(v[Rank], v[Core]).second)
                throw std::runtime_error("Invalid or duplicate Perfetto core record");
            for (uint32_t w = 1; w < StageRows; ++w)
                if (r.present[w] && v[w] > v[FlushDone]) throw std::runtime_error("Timestamp outside profile body");
            const auto pid = v[Rank] + 1, tid = v[Core] * 64 + 1;
            if (ranks.insert(v[Rank]).second)
                metadata("process_name", pid, 0, "Rank " + std::to_string(v[Rank]) + " / device " +
                         std::to_string(r.device) + " / independent core origins");
            const bool worker = v[CoalesceWorker] || r.present[CoalesceSendBegin];
            const auto role = v[OpsRole] ? (v[OpsRole] == 1 ? "OPS SCATTER" :
                             v[OpsRole] == 2 ? "OPS GATHER" : "OPS RDMA") :
                             worker ? "COALESCE" : (r.receive ? "RECEIVE" : "SEND");
            const auto label = "AIV " + std::to_string(v[Core]) + " " + role;
            metadata("thread_name", pid, tid, label + " / stages (own entry = 0)");
            auto event = [&](const std::string& name, uint64_t begin, uint64_t end, uint64_t track, bool instant) {
                item(); out << "{\"ph\":\"" << (instant ? "i" : "X") << "\",\"cat\":\"ll_combine\",\"name\":"
                    << trace_quote(name) << ",\"pid\":" << pid << ",\"tid\":" << track << ",\"ts\":" << us(begin);
                if (instant) out << ",\"s\":\"t\""; else out << ",\"dur\":" << us(end - begin);
                out << ",\"args\":{\"begin_offset_ticks\":" << begin << ",\"end_offset_ticks\":" << end;
            };
            event("profiled body", 0, v[FlushDone], tid, false);
            out << ",\"entry_tick\":" << trace_quote(std::to_string(v[Entry])) << ",\"note\":" << trace_quote(note);
            for (uint32_t w = StageRows; w < Count; ++w)
                if (r.present[w] && w != Valid) out << ',' << trace_quote(fields[w]) << ':' << v[w];
            out << "}}";
            std::vector<Slice> slices;
            auto add = [&](const std::string& name, Word a, Word b) {
                if (!r.present[a] || !r.present[b]) return;
                const uint64_t start = a == Entry ? 0 : v[a];
                if (v[b] < start) throw std::runtime_error("Non-monotonic Perfetto stage: " + name);
                if (v[b] > start) slices.push_back({name, start, v[b]});
            };
            if (v[OpsRole]) {
                add("profile setup", Entry, ProfileInitEnd);
                add("ops initialization", ProfileInitEnd, InitEnd);
                add("ops local scatter", StageBegin, StageEnd);
                add("ops local gather and reduction", PartialBegin, PartialEnd);
                add("ops RDMA sender (includes readiness and quiet)", DirectBegin, DirectEnd);
                add(v[OpsPolicy] >= 2 ? "ops exit all-core barrier" : "ops all-core barrier", ReadyBegin, ReadyEnd);
                add("ops deferred transport retirement", ReadyRetireBegin, ReadyRetireEnd);
                add("ops final combine", FinalBegin, FinalEnd);
                add("ops final setup", FinalBegin, FinalInitEnd);
            } else for (auto s : spans) {
                // Task readiness is interleaved with useful partial work. Do
                // not render that entire observation interval as blocking wait.
                if(v[ReadyTaskRounds] && s.begin==ReadyBegin) {
                    if(s.end==ReadyObserved) add("group readiness observation window (includes work)",s.begin,s.end);
                } else if(v[ReadyTaskRounds] && s.begin==ReadyObserved && s.end==ReadyEnd) {
                    continue; // retirement has its own explicit interval
                } else add(s.name,s.begin,s.end);
            }
            if (worker) {
                // WaitProducers has an end stamp only. Label the surrounding
                // window honestly: it also includes the preceding PIPE_ALL.
                add("coalesce producer readiness window (incl. setup)", r.present[StageEnd] ? StageEnd : InitEnd,
                    CoalesceReady);
                add("coalesce pipe tail", CoalesceReceiveEnd, PipeDone);
            }
            else add("pipe tail", r.receive ? DrainEnd : DirectEnd, PipeDone);
            // SMALL_BATCH may wait for group ready within partial processing.
            // Put crossing intervals on separate lanes; never clip or stack them.
            std::sort(slices.begin(), slices.end(), [](const Slice& a, const Slice& b) {
                return a.begin != b.begin ? a.begin < b.begin : a.end > b.end;
            });
            std::vector<std::vector<uint64_t>> ends(1);
            for (auto& s : slices) {
                for (;; ++s.lane) {
                    if (s.lane == ends.size()) {
                        ends.emplace_back();
                        metadata("thread_name", pid, tid + s.lane, label + " / overlapping stage " + std::to_string(s.lane));
                    }
                    auto& stack = ends[s.lane];
                    while (!stack.empty() && stack.back() <= s.begin) stack.pop_back();
                    if (stack.empty() || s.end <= stack.back()) { stack.push_back(s.end); break; }
                }
                if (s.lane >= 63) throw std::runtime_error("Too many overlapping profile stages");
                event(s.name, s.begin, s.end, tid + s.lane, false); out << "}}";
            }
            metadata("thread_name", pid, tid + 63, label + " / observations");
            for (uint32_t w = 1; w < StageRows; ++w)
                if (r.present[w]) { event(fields[w], v[w], v[w], tid + 63, true); out << "}}"; }
            if (r.present[TailObservations] && v[TailObservations])
                for (auto w : {TailFirstSeen, TailLastSeen, TailReady}) {
                    if (w == TailReady && !v[w]) continue;
                    if (v[w] > v[FlushDone]) throw std::runtime_error("Tail observation outside profile body");
                    event(fields[w], v[w], v[w], tid + 63, true);
                    for (auto detail : {TailToken, TailSlot, TailBlock, TailMissingMask, TailMissingBlocks, TailObservations,
                                        TailSampleStride, TailInspectCalls, TailInspectSkipped})
                        out << ',' << trace_quote(fields[detail]) << ':' << v[detail];
                    out << "}}";
                }
        }
        out << "\n]}\n";
        out.close();
        if (!out) throw std::runtime_error("Cannot write Perfetto trace");
        // Sample filenames are unique within a run; never truncate an existing trace.
        if (std::filesystem::exists(path)) throw std::runtime_error("Perfetto output already exists: " + path.string());
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
} // namespace llbench
