// Included by the offline reporter after its CSV and profile record helpers.
struct SuiteResult {
    std::string variant, kind;
    std::map<std::string, std::string> config;
    std::map<uint32_t, Event> events;
    std::map<uint32_t, std::vector<Record>> records;
    struct Producer { uint32_t rank, core, type, row; std::string path; };
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Producer> producers;
};
const Record* receive_tail(const SuiteResult& d, uint32_t i)
{
    const Record* tail = nullptr;
    const auto it = d.records.find(i);
    if (it == d.records.end()) return nullptr;
    for (const auto& r : it->second)
        if (r.receive && r[Rank] == d.events.at(i).rank && (!tail || r[FlushDone] > (*tail)[FlushDone])) tail = &r;
    return tail;
}
void tail_producers(const SuiteResult& d, const Record& r)
{
    if (!r[TailObservations]) return;
    std::cout << " T t=" << r[TailToken] << " first=k" << r[TailSlot] << "/b" << r[TailBlock]
              << " blocks/seen=" << r[TailMissingBlocks] << '/' << r[TailObservations]
              << " time=" << r[TailFirstSeen] << '/' << r[TailLastSeen] << '/' << r[TailReady]
              << " src=";
    for (uint32_t slot = 0; slot < 32; ++slot) if ((r[TailMissingMask] >> slot) & 1U) {
        const auto found = d.producers.find({uint32_t(r[Rank]), uint32_t(r[TailToken]), slot});
        if (found == d.producers.end()) { std::cout << slot << ":unknown "; continue; }
        const auto& p = found->second;
        std::cout << slot << ':' << p.rank << '/' << p.core << '/' << p.path
                  << '/' << (p.type == 1 ? "D" : "P") << ' ';
    }
    std::cout << '\n';
}
void tail_stability(const SuiteResult& d)
{
    using Key = std::tuple<uint32_t, uint32_t, std::string, uint32_t>;
    std::map<Key, uint32_t> count;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> packets;
    uint32_t observed = 0;
    for (const auto& [i, e] : d.events) {
        const auto* r = receive_tail(d, i);
        if (!r || !(*r)[TailObservations]) continue;
        ++observed;
        std::set<Key> unique;
        for (uint32_t slot = 0; slot < 32; ++slot) if (((*r)[TailMissingMask] >> slot) & 1U) {
            const auto key = std::make_tuple(uint32_t((*r)[Rank]), uint32_t((*r)[TailToken]), slot);
            const auto found = d.producers.find(key);
            if (found == d.producers.end()) continue;
            const auto& p = found->second;
            unique.emplace(p.rank, uint32_t((*r)[Rank]), p.path, p.type);
            ++packets[key];
        }
        for (const auto& key : unique) ++count[key];
    }
    std::vector<std::pair<Key, uint32_t>> sorted(count.begin(), count.end());
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::cout << " Tfreq " << d.variant << " event-tail-missed=" << observed << '/' << d.events.size()
              << " links=";
    for (size_t j = 0; j < std::min<size_t>(4, sorted.size()); ++j) {
        const auto& [key, n] = sorted[j];
        std::cout << std::get<0>(key) << '>' << std::get<1>(key) << '/' << std::get<2>(key) << '/'
                  << (std::get<3>(key) == 1 ? "D" : "P") << ':' << n << ' ';
    }
    const auto most = std::max_element(packets.begin(), packets.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    if (most != packets.end()) {
        const auto& [key, n] = *most;
        std::cout << " same-slot=r" << std::get<0>(key) << "/t" << std::get<1>(key) << "/k" << std::get<2>(key) << ':' << n;
    }
    std::cout << '\n';
}
double percentile(const std::vector<double>& sorted, double fraction)
{
    const double index = fraction * (sorted.size() - 1);
    const auto low = size_t(index), high = size_t(std::ceil(index));
    return sorted[low] + (index - low) * (sorted[high] - sorted[low]);
}
std::string compact_cost_name(Word w)
{
    std::string name = names[w];
    const std::string suffix = "_ticks";
    if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
        name.resize(name.size() - suffix.size());
    for (const auto& prefix : {std::make_pair("stage_", "s."), std::make_pair("direct_", "d."), std::make_pair("partial_", "m4."),
                               std::make_pair("final_", "m5."), std::make_pair("ready_", "g.")})
        if (name.rfind(prefix.first, 0) == 0) return prefix.second + name.substr(std::string(prefix.first).size());
    return name;
}
void compact_times(const Record& r, std::initializer_list<Word> words)
{
    bool first = true;
    for (Word w : words) {
        std::cout << (first ? "" : "/");
        if (r.present[w]) std::cout << r[w];
        else std::cout << '-';
        first = false;
    }
}
void compact_record(const Record& r)
{
    if(r[CoalesceWorker]) {
        std::cout << " A r/c=" << r[Rank] << '/' << r[Core] << " body=" << r[FlushDone] << " end";
        for(Word w : {CoalesceReady,CoalesceSendBegin,RdmaFirstPut,RdmaLastPut,CoalesceSendEnd,
                      CoalesceFirstReady,CoalesceLastReady,CoalesceReceiveEnd})
            if(r.present[w]) std::cout << ' ' << names[w] << '=' << r[w];
        std::cout << "\n  pack wait/gather/put/quiet/recv_wait/unpack=" << r[CoalesceWaitTicks] << '/'
                  << r[CoalescePackTicks] << '/' << r[RdmaPutTicks] << '/' << r[RdmaQuietTicks] << '/'
                  << r[CoalesceReceiveWaitTicks] << '/' << r[CoalesceUnpackTicks]
                  << " rows/wqes/bytes/unpack/forward=" << r[CoalesceRows] << '/' << r[RdmaPackets] << '/'
                  << r[CoalesceBytes] << '/' << r[CoalesceUnpackRows] << '/' << r[CoalesceForwardRows] << '\n';
        return;
    }
    std::cout << (r.receive ? " R" : " S") << " r/c=" << r[Rank] << '/' << r[Core]
              << " body=" << r[FlushDone] << " rows=" << r[StageRows] << '/' << r[DirectRows] << '/' << r[PartialRows]
              << " final=" << r[FinalTokens] << '/' << r[Contributions] << " local/get=" << r[PartialLocalContrib]
              << '/' << r[PartialGetContrib] << '\n';
    std::cout << "  e=";
    if (r.receive)
        compact_times(r, {InitEnd, StageDrained, CoreNotifyEnd, ReadyBegin, ReadyObserved, ReadyEnd, PartialEnd,
                           RdmaFirstPut, RdmaLastPut, FinalFirstReady, FinalLastReady, FinalLastStore, DrainEnd, PipeDone});
    else
        compact_times(r, {InitEnd, StageEnd, CoreNotifyEnd, NotifyEnd, DirectFirstSubmit, DirectLastSubmit, DirectEnd, PipeDone});
    std::cout << "\n  top";
    const auto costs = top_costs(r, kDetailBegin, kDetailEnd);
    for (size_t j = 0; j < std::min<size_t>(4, costs.size()); ++j)
        std::cout << ' ' << compact_cost_name(costs[j]) << '=' << r[costs[j]];
    std::cout << '\n';
    // Always expose staging/Put/CQ and the moved submission cost. A top-N list
    // alone can hide them when other waits become longer.
    if (r[RdmaPackets] || r[RdmaQuietTicks])
        std::cout << "  rdma=" << r[RdmaPackets] << '/' << r[RdmaDeferredPackets]
                  << '/' << r[RdmaDeferGapTicks] << '/' << r[RdmaStageIssueTicks] << '/' << r[RdmaStageWaitTicks]
                  << '/' << r[RdmaPutTicks] << '/' << r[RdmaQuietTicks] << '\n';
    if (r.receive) {
        std::cout << "  probe=" << r[ProbeMisses] << '/' << r[ProbeCalls]
                  << '/' << r[ProbeMissTicks] << '/' << r[ProbeFlagBlocks] << '/' << r[ProbeConfirmedSlots];
        if (r.present[ProbeBlockedSlot] && r[ProbeMisses]) {
            std::cout << " blocked=t" << r[ProbeBlockedToken] << "/k";
            if (r[ProbeBlockedSlot] == 0xffffffffU) std::cout << "all";
            else std::cout << r[ProbeBlockedSlot];
        }
        std::cout << " inspect=" << r[TailInspectTicks] << '\n';
    }
    if (r[EarlyIssueTicks] || r[MetadataBatches] || r[PollPauseCalls])
        std::cout << "  extra early/meta/pause_calls/pause_ticks=" << r[EarlyIssueTicks] << '/' << r[MetadataBatches]
                  << '/' << r[PollPauseCalls] << '/' << r[PollPauseTicks] << '\n';
    if (r[ReadyPollCalls]) {
        std::cout << "  g=" << r[ReadyPollMisses] << '/' << r[ReadyPollCalls]
                  << '/' << r[ReadyPollMaxTicks] << '/' << r[ReadyInspectTicks] << ' ';
        if (r[ReadyMissingCount]) std::cout << r[ReadyMissingRank] << '/' << r[ReadyMissingCore] << '/' << r[ReadyMissingCount];
        else std::cout << '-';
        std::cout << '\n';
    }
    if (r.receive && (r[ReadyTaskRounds] || r[ReadySnapshotTicks]))
        std::cout << "  q=" << r[ReadyFirstTask] << '/' << r[ReadyObserved] << '/' << r[ReadyTasksDone]
                  << ' ' << r[ReadyTaskRounds] << '/' << r[ReadyTaskStalls] << '/' << r[ReadyFilterSkips]
                  << ' ' << r[ReadyTasksBeforeAll] << '/' << r[ReadyRemoteBeforeAll]
                  << ' ' << r[ReadyCachedTasks] << '/' << r[ReadyMetadataReloads]
                  << ' ' << r[ReadySetupTicks] << '/' << r[ReadyPlanTicks] << '/' << r[ReadyFilterTicks]
                  << '/' << r[ReadySnapshotTicks] << '/' << r[ReadyPipelineIssueTicks]
                  << ' ' << (r[ReadyRetireEnd]-r[ReadyRetireBegin]) << '\n';
    if (r[ReadyAckWaitTicks] || r[ReadyConsumedWaitTicks])
        std::cout << "  ack/consumed=" << r[ReadyAckWaitTicks] << '/' << r[ReadyConsumedWaitTicks] << '\n';
    if (r.receive && (r[RouteLoadBatches] || r[RouteSnapshotCapacity])) {
        std::cout << "  scan=" << r[RouteLoadBatches] << '/' << r[RouteLoadedRows] << '/' << r[RouteSelectPasses];
        if (r[RouteSnapshotCapacity])
            std::cout << " ub=" << r[RouteSnapshotBytes] << '/' << r[RouteSnapshotCapacity] << '/' << r[RouteSnapshotFallbacks];
        std::cout << '\n';
    }
    if (r.receive && r[BatchGroupCalls] + r[BatchWholeTokens])
        std::cout << "  batch group_checks/whole_tokens/route_ticks=" << r[BatchGroupCalls] << '/'
                  << r[BatchWholeTokens] << '/' << r[BatchRouteTicks] << '\n';
    if (r.receive && r[LocalContributions] + r[RemoteContributions]) {
        std::cout << "  final_by_host local/remote contributions=" << r[LocalContributions] << '/' << r[RemoteContributions]
                  << " probe_ticks=" << r[LocalProbeTicks] << '/' << r[RemoteProbeTicks]
                  << " misses=" << r[LocalProbeMisses] << '/' << r[RemoteProbeMisses]
                  << " calls=" << r[LocalProbeCalls] << '/' << r[RemoteProbeCalls]
                  << " reduce_ticks=" << r[LocalReduceTicks] << '/' << r[RemoteReduceTicks] << '\n';
    }
}
void suite_report(const fs::path& root)
{
    std::vector<fs::path> directories;
    for (const auto& f : fs::directory_iterator(root))
        if (f.is_directory() && fs::exists(f.path() / "summary.csv")) directories.push_back(f.path());
    std::sort(directories.begin(), directories.end());
    if (directories.empty()) throw std::runtime_error("No completed phase summaries in " + root.string());
    std::map<uint32_t, std::vector<SuiteResult>> cases;
    for (const auto& directory : directories) {
        Csv csv(directory / "summary.csv");
        std::map<uint32_t, SuiteResult> data;
        while (csv.next()) {
            const auto bs = uint32_t(integer(csv.get("tokens")));
            auto& d = data[bs];
            if (!d.config.empty()) throw std::runtime_error("Duplicate case summary");
            d.variant = directory.filename().string();
            for (size_t i = 0; i < csv.header.size(); ++i) d.config[csv.header[i]] = csv.cells[i];
            d.kind = csv.column("profile_build") >= 0 && csv.get("profile_build") != "0" ? "profile" : "clean";
        }
        Csv events(directory / "rank-max.csv");
        while (events.next()) {
            auto it = data.find(uint32_t(integer(events.get("tokens"))));
            if (it == data.end()) throw std::runtime_error("Event has no case summary");
            const auto i = uint32_t(integer(events.get("iteration"))), rank = uint32_t(integer(events.get("rank")));
            const double us = std::stod(events.get("us"));
            if (!std::isfinite(us) || us <= 0 || rank >= integer(it->second.config.at("ep")) ||
                !it->second.events.emplace(i, Event{us, rank}).second) throw std::runtime_error("Invalid global Event");
        }
        if (fs::exists(directory / "diagnostic.csv")) {
            Csv diag(directory / "diagnostic.csv");
            std::array<int, Count> columns;
            for (uint32_t w = 0; w < Count; ++w) columns[w] = diag.column(names[w]);
            while (diag.next()) {
                Record r;
                r.iteration = uint32_t(integer(diag.get("iteration")));
                r.device = uint32_t(integer(diag.get("device")));
                r.receive = diag.get("role") == "receive";
                for (uint32_t w = 0; w < Count; ++w)
                    if (columns[w] >= 0 && !diag.cells[columns[w]].empty()) {
                        r.v[w] = integer(diag.cells[columns[w]]);
                        r.present[w] = true;
                    }
                auto it = data.find(uint32_t(r[Tokens]));
                if (it == data.end() || !it->second.events.count(r.iteration) || !r[Complete] ||
                    !r.present[FlushDone] || r[Rank] >= r[World]) throw std::runtime_error("Invalid diagnostic record");
                it->second.records[r.iteration].push_back(r);
            }
        }
        if (fs::exists(directory / "producers.csv")) {
            Csv producers(directory / "producers.csv");
            while (producers.next()) {
                const auto bs = uint32_t(integer(producers.get("tokens")));
                auto& d = data.at(bs);
                const auto key = std::make_tuple(uint32_t(integer(producers.get("source"))),
                    uint32_t(integer(producers.get("token"))), uint32_t(integer(producers.get("slot"))));
                SuiteResult::Producer p{uint32_t(integer(producers.get("producer"))), uint32_t(integer(producers.get("core"))),
                    uint32_t(integer(producers.get("type"))), uint32_t(integer(producers.get("row"))), producers.get("path")};
                if (!d.producers.emplace(key, p).second) throw std::runtime_error("Duplicate return producer");
            }
        }
        for (auto& [bs, d] : data) {
            if (d.events.size() != integer(d.config.at("samples"))) throw std::runtime_error("Missing global samples");
            if (d.kind == "profile" && d.config.count("profile_level") &&
                integer(d.config.at("profile_level")) && d.records.empty())
                throw std::runtime_error("Missing global diagnostics");
            uint32_t expected = 0;
            for (const auto& [i, e] : d.events)
                if (i != expected++) throw std::runtime_error("Non-contiguous global samples");
            cases[bs].push_back(std::move(d));
        }
    }
    std::cout << "LL_COMBINE compact=v2 global rank-MAX; Event=us; profile=ticks/core-entry; no samples removed\n"
                 "IQR=candidates; WIDE=(max-min)/P50>20%; full samples/fields in CSV; '-'=absent.\n"
                 "e(S)=init/stage/core_notify/notify/direct_first/direct_last/direct_end/pipe\n"
                 "e(R)=init/stage_drained/core_notify/ready_begin/ready_observed/ready_end/partial_end/put_first/put_last/final_first/final_last/store/drain/pipe\n"
                 "top4 cumulative ticks: s.=stage_, d.=direct_, m4.=partial_, m5.=final_, g.=ready_; overlapping costs are not additive.\n"
                 "rdma=n/deferred/gap/stage_issue/stage_wait/put/quiet\n"
                 "probe=miss/calls/miss_ticks/flags/confirmed; N=r/c local_wait/publish miss/calls/max_probe missing_core/count stores inspect\n"
                 "P=own-entry rank/core:first_put/last_put/partial_end/packets\n"
                 "g=miss/calls/max_probe/inspect missing_rank/core/count; 4294967295=all\n"
                 "G=r/c window=W(wait)|O(observe, includes work) miss/calls/max_probe missing_rank/core/count inspect; omitted if same as R\n"
                 "q=first/all/done rounds/stalls/skips before_all/remote cached/reloads setup/plan/filter/snapshot/pipeline_issue retire\n"
                 "scan=GM_batches/rows/Vector_group_passes (16B/row); ub=snapshot_bytes/capacity/fallbacks\n"
                 "T=last-miss; time=first/last/ready; src=slot:rank/core/path/type; D=direct P=partial.\n"
                 "Tfreq links=producer>receiver/path/type:sample_count\n";
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& [bs, results] : cases) {
        const auto& config = results.front().config;
        std::cout << "\nBS=" << bs;
        for (const char* key : {"ep", "hidden", "topk", "experts", "dtype", "entry_sync", "warmup", "route", "seed"})
            std::cout << ' ' << key << '=' << config.at(key);
        std::cout << "\nvariant                 mode      N      mean       p50       min       p95       max     CV%\n";
        for (const auto& d : results) {
            for (const char* key : {"ep", "hidden", "topk", "experts", "dtype", "entry_sync", "warmup", "route", "seed"})
                if (config.at(key) != d.config.at(key)) throw std::runtime_error("Mixed case configuration");
            std::vector<double> values;
            double total = 0, square = 0;
            for (const auto& [i, e] : d.events) { values.push_back(e.us); total += e.us; }
            const double mean = total / values.size();
            for (auto v : values) square += (v - mean) * (v - mean);
            std::sort(values.begin(), values.end());
            const double p50 = percentile(values, .5);
            std::cout << std::left << std::setw(24) << d.variant << ' ' << std::setw(8) << d.kind << std::right
                      << std::setw(4) << values.size() << std::setw(10) << mean << std::setw(10) << p50
                      << std::setw(10) << values.front() << std::setw(10) << percentile(values, .95)
                      << std::setw(10) << values.back() << std::setw(8) << std::sqrt(square / values.size()) / mean * 100
                      << ((values.back() - values.front()) > .2 * p50 ? " WIDE" : "") << '\n';
        }
        for (const auto& d : results) {
            std::vector<double> values;
            for (const auto& [i, e] : d.events) values.push_back(e.us);
            std::sort(values.begin(), values.end());
            const double q1 = percentile(values, .25), q3 = percentile(values, .75), limit = q3 + 1.5 * (q3 - q1);
            std::vector<std::pair<uint32_t, Event>> flagged;
            for (const auto& [i, e] : d.events) if (e.us > limit) flagged.push_back({i, e});
            if (!flagged.empty()) {
                std::cout << "IQR " << d.variant << " n=" << flagged.size() << " limit=" << limit << " i/r/us=";
                for (size_t j = 0; j < std::min<size_t>(8, flagged.size()); ++j)
                    std::cout << (j ? "," : "") << flagged[j].first << '/' << flagged[j].second.rank << '/' << flagged[j].second.us;
                std::cout << '\n';
            }
            if (d.records.empty()) continue;
            if (!d.producers.empty()) tail_stability(d);
            std::vector<std::pair<uint32_t, Event>> order(d.events.begin(), d.events.end());
            std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second.us < b.second.us; });
            const auto median = order[(order.size() - 1) / 2].first, worst = order.back().first;
            for (auto i : std::set<uint32_t>{median, worst}) {
                const auto& event = d.events.at(i);
                const auto& records = d.records.at(i);
                std::set<uint32_t> ranks;
                for (const auto& r : records) ranks.insert(uint32_t(r[Rank]));
                if (ranks.size() != integer(d.config.at("ep"))) throw std::runtime_error("Partial global diagnostics");
                std::cout << "D " << d.variant << ' ' << (i == worst ? "worst" : "p50") << " i=" << i
                          << " event_r=" << event.rank << " us=" << event.us << '\n';
                const Record* receiveRecord = nullptr;
                for (bool receive : {false, true}) {
                    const Record* tail = nullptr;
                    for (const auto& r : records)
                        if (!r[CoalesceWorker] && r[Rank] == event.rank && r.receive == receive &&
                            (!tail || r[FlushDone] > (*tail)[FlushDone])) tail = &r;
                    if (!tail) throw std::runtime_error("Missing Event rank diagnostic role");
                    compact_record(*tail);
                    if (receive) { receiveRecord = tail; tail_producers(d, *tail); }
                }
                const Record* notifier=nullptr;
                const Record* groupWait=nullptr;
                for(const auto& r:records) if(r[Rank]==event.rank && r.present[ReadyObserved] &&
                    (!groupWait || r[ReadyObserved]-r[ReadyBegin]>(*groupWait)[ReadyObserved]-(*groupWait)[ReadyBegin])) groupWait=&r;
                // The same record's ready window is already present in e(R).
                if(groupWait && groupWait != receiveRecord) {
                    const auto& r=*groupWait;
                    std::cout << " G " << r[Rank] << '/' << r[Core] << (r[ReadyTaskRounds]?" O=":" W=") << r[ReadyObserved]-r[ReadyBegin]
                              << ' ' << r[ReadyPollMisses] << '/' << r[ReadyPollCalls] << '/' << r[ReadyPollMaxTicks]
                              << ' ' << r[ReadyMissingRank] << '/' << r[ReadyMissingCore] << '/' << r[ReadyMissingCount]
                              << ' ' << r[ReadyInspectTicks] << '\n';
                }
                for(const auto& r:records) if(r[Rank]==event.rank && r.present[NotifyBegin] &&
                    (!notifier || r[NotifyEnd]-r[NotifyBegin]>(*notifier)[NotifyEnd]-(*notifier)[NotifyBegin])) notifier=&r;
                if(notifier) {
                    const auto& r=*notifier;
                    std::cout << " N " << r[Rank] << '/' << r[Core]
                              << ' ' << r[NotifyLocalReady]-r[NotifyBegin] << '/' << r[NotifyEnd]-r[NotifyLocalReady]
                              << ' ' << r[NotifyPollMisses] << '/' << r[NotifyPollCalls] << '/' << r[NotifyPollMaxTicks]
                              << ' ' << r[NotifyMissingCore] << '/' << r[NotifyMissingCount]
                              << ' ' << r[NotifyStores] << ' ' << r[NotifyInspectTicks] << '\n';
                }
                const Record* worker=nullptr;
                for(const auto& r:records) if(r[Rank]==event.rank && r[CoalesceWorker] &&
                    (!worker || r[FlushDone]>(*worker)[FlushDone])) worker=&r;
                if(worker) compact_record(*worker);
                std::vector<const Record*> puts;
                for (const auto& r : records) if (r.present[RdmaLastPut]) puts.push_back(&r);
                std::stable_sort(puts.begin(), puts.end(), [](auto a, auto b) { return (*a)[RdmaLastPut] > (*b)[RdmaLastPut]; });
                std::cout << " P ";
                for (size_t j = 0; j < std::min<size_t>(2, puts.size()); ++j) {
                    const auto& r = *puts[j];
                    std::cout << r[Rank] << '/' << r[Core] << ':' << r[RdmaFirstPut] << '/' << r[RdmaLastPut]
                              << '/' << r[PartialEnd] << '/' << r[RdmaPackets] << ' ';
                }
                std::cout << '\n';
            }
        }
        std::cout << "END BS=" << bs << '\n';
    }
    std::cout << "LL_COMBINE END cases=" << cases.size() << '\n';
}
