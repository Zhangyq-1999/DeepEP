// Included by the offline reporter after its CSV and profile record helpers.
struct SuiteResult {
    std::string variant, kind;
    std::map<std::string, std::string> config;
    std::map<uint32_t, Event> events;
    std::map<uint32_t, std::vector<Record>> records;
    struct Producer { uint32_t rank, core, type, row; std::string path; };
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Producer> producers;
};
std::string diagnostic_id(const std::string& variant)
{
    const auto dash = variant.find('-');
    if (dash != std::string::npos && dash && variant.substr(0, dash).find_first_not_of("0123456789") == std::string::npos)
        return variant.substr(0, dash);
    return variant;
}
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
    std::cout << " Tfreq " << diagnostic_id(d.variant) << " event-tail-missed=" << observed << '/' << d.events.size()
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
void compact_record(const Record& r, bool brief = false)
{
    if (r[OpsRole]) {
        const char* role = r[OpsRole] == 1 ? "scatter" : r[OpsRole] == 2 ? "gather" : "rdma";
        std::cout << " O r/c=" << r[Rank] << '/' << r[Core] << " role=" << role << " body=" << r[FlushDone]
                  << " rows=" << r[StageRows] << '/' << r[PartialRows]
                  << " contrib=" << r[PartialLocalContrib] << '/' << r[Contributions]
                  << " final=" << r[FinalTokens] << " active=";
        compact_times(r, {ActiveStageCores, ActiveGatherCores, ActiveRdmaCores, ActiveFinalCores});
        std::cout << " policy=" << r[OpsPolicy] << " retire=";
        compact_times(r, {ReadyRetireBegin, ReadyRetireEnd});
        std::cout << " deferred/peers=" << r[OpsDeferredCommits] << '/' << r[OpsRetiredPeers];
        std::cout << "\n  oe=";
        compact_times(r, {InitEnd, StageEnd, ReadyFirstTask, PartialEnd, RdmaFirstPut, RdmaLastPut, DirectEnd,
                          ReadyBegin, ReadyEnd, FinalInitEnd, FinalFirstReady, FinalLastReady, FinalLastStore, FinalEnd});
        if (r[Level] >= 2)
            std::cout << "\n  op plan/rows/tasks=" << r[ReadyPlanTicks] << '/' << r[RouteLoadedRows] << '/' << r[ReadyCachedTasks]
                      << " poll= " << r[ReadyPollMisses] << '/' << r[ReadyPollCalls] << '/' << r[ReadyPollCheckTicks]
                      << " tx=n/bytes/put/quiet/calls/max:" << r[RdmaPackets] << '/' << r[RdmaBytes] << '/' << r[RdmaPutTicks]
                      << '/' << r[RdmaQuietTicks] << '/' << r[RdmaQuietCalls] << '/' << r[RdmaQuietMaxTicks]
                      << " final_poll=" << r[ProbeMisses] << '/' << r[ProbeCalls] << '/' << r[ProbeTicks]
                      << " reduce_store=" << r[PartialComputeIssueTicks] << '/' << r[PartialPackTicks]
                      << '/' << r[FinalComputeIssueTicks] << '/' << r[FinalStoreIssueTicks];
        // Queue operations only execute on the RDMA workers. Keep the complete
        // fields in CSV/Perfetto, without repeating empty queue rows for gather.
        if (r[Level] >= 2 && r[OpsPolicy] >= 3 && r[OpsRole] == 3)
            std::cout << "\n  oq batch=n/max/copies:" << r[OpsBatchCommits] << '/' << r[OpsBatchMax]
                      << '/' << r[OpsSqCopies] << " build=" << r[WqePrepareTicks]
                      << " cq=calls/loads/entries/ticks/capacity:" << r[OpsCqPollCalls] << '/' << r[OpsCqLoads]
                      << '/' << r[OpsCqEntries] << '/' << r[OpsCqTicks] << '/' << r[OpsCapacityPolls];
        std::cout << '\n';
        return;
    }
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
    if (brief && !r.receive) {
        std::cout << " S r/c=" << r[Rank] << '/' << r[Core] << " body=" << r[FlushDone] << " send_end=";
        compact_times(r, {DirectLastSubmit, DirectEnd, PipeDone});
        std::cout << '\n';
        return;
    }
    std::cout << (r.receive ? " R" : " S") << " r/c=" << r[Rank] << '/' << r[Core]
              << " body=" << r[FlushDone] << " rows=" << r[StageRows] << '/' << r[DirectRows] << '/' << r[PartialRows]
              << " final=" << r[FinalTokens] << '/' << r[Contributions] << " lg=" << r[PartialLocalContrib]
              << '/' << r[PartialGetContrib] << '\n';
    std::cout << "  e=";
    if (r.receive)
        compact_times(r, {InitEnd, StageDrained, CoreNotifyEnd, ReadyBegin, ReadyObserved, ReadyEnd, PartialEnd,
                           RdmaFirstPut, RdmaLastPut, FinalFirstReady, FinalLastReady, FinalLastStore, DrainEnd, PipeDone});
    else
        compact_times(r, {InitEnd, StageEnd, CoreNotifyEnd, NotifyEnd, DirectFirstSubmit, DirectLastSubmit, DirectEnd, PipeDone});
    if (r.receive && r[Level] >= 2)
        std::cout << "\n  m4=" << r[PartialLocalIssueTicks] << '/' << r[PartialGetIssueTicks]
                  << '/' << r[PartialLocalWaitTicks] << '/' << r[PartialGetWaitTicks]
                  << '/' << r[PartialComputeIssueTicks] << '/' << r[PartialOutputWaitTicks]
                  << '/' << r[PartialPackTicks] << '/' << r[PartialInputDrainTicks];
    if (r[Level] >= 2 && !brief) {
        std::cout << "\n  cache=" << r[DataL2Policy]
                  << " m3=" << r[StageInputTicks] << '/' << r[StageStoreIssueTicks] << '/' << r[StageDrainTicks]
                  << " g=" << r[ReadyPollLoadTicks] << '/' << r[ReadyPollCheckTicks]
                  << '/' << r[ReadyPollCalls]
                  << " m5=" << r[FinalInputIssueTicks] << '/' << r[FinalInputWaitTicks]
                  << '/' << r[FinalComputeIssueTicks] << '/' << r[FinalStoreIssueTicks];
    }
    if (r[Level] >= 2 && r.present[PartialOutputInitLanes]) {
        if (r.receive)
            std::cout << "\n  small=" << r[PartialOutputInitLanes] << '/' << r[PartialSetupTicks]
                      << '/' << r[ReadyMetadataWaitTicks] << '/' << r[ReadySingleTasks];
        else
            std::cout << "\n  direct_meta=" << r[DirectMetadataCachedRows] << '/'
                      << r[DirectMetadataReloads] << '/' << r[DirectActiveLanes];
        if (!r.receive && r.present[DirectRoutePreloadRows])
            std::cout << " route_plan=" << r[DirectRoutePreloadRows] << '/'
                      << r[DirectRoutePreloadWaitTicks] << '/' << r[DirectRouteGatherRows];
    }
    if (!r.receive && r[DirectLocalDeferredRows])
        std::cout << " priority=" << r[DirectLocalDeferredRows] << '/' << r[DirectPriorityScanTicks];
    if (r[TinyPolicy]) std::cout << " tiny=" << r[TinyPolicy] << '/' << r[FinalSparseProbeCalls]
            << '/' << r[FinalSparseConfirmedSlots] << '/' << r[FinalSparseSkippedFlags];
    if (r[RdmaWaitSplitCalls])
        std::cout << "\n  txwait=" << r[RdmaPrefixWaitTicks] << '/' << r[RdmaCopyWaitTicks]
                  << '/' << r[RdmaWaitSplitCalls];
    if (brief) {
        std::cout << "\n  rdma=" << r[RdmaPackets] << '/' << r[RdmaDeferredPackets]
                  << '/' << r[RdmaDeferGapTicks] << '/' << r[RdmaStageIssueTicks] << '/' << r[RdmaStageWaitTicks]
                  << '/' << r[RdmaPutTicks] << '/' << r[RdmaQuietTicks];
        if (r.receive && r[PartialInputLanes])
            std::cout << " input=" << r[PartialInputLanes] << '/' << r[ReadyInputIssued] << '/' << r[ReadyInputUsed];
        std::cout << '\n';
        return;
    }
    std::cout << "\n  top";
    const auto costs = top_costs(r, kDetailBegin, kDetailEnd);
    for (size_t j = 0; j < std::min<size_t>(4, costs.size()); ++j)
        std::cout << ' ' << compact_cost_name(costs[j]) << '=' << r[costs[j]];
    std::cout << '\n';
    // Always expose staging/Put/CQ and the moved submission cost. A top-N list
    // alone can hide them when other waits become longer.
    if (r[RdmaPackets] || r[RdmaQuietTicks]) {
        std::cout << "  rdma=" << r[RdmaPackets] << '/' << r[RdmaDeferredPackets]
                  << '/' << r[RdmaDeferGapTicks] << '/' << r[RdmaStageIssueTicks] << '/' << r[RdmaStageWaitTicks]
                  << '/' << r[RdmaPutTicks] << '/' << r[RdmaQuietTicks];
        if (r.receive && r.present[RdmaStageWaitCalls])
            std::cout << " stage=" << r[RdmaStageWaitCalls] << '/' << r[RdmaStagePairGroups]
                      << '/' << r[PartialPrefetchSubmitCalls];
        if (r.receive && (r[PartialPackedSubmitCalls] || r[PartialEagerPackets]))
            std::cout << " stream=" << r[PartialPackedSubmitCalls] << '/' << r[PartialEagerPackets]
                      << '/' << r[ReadyPipelineIssueTicks];
        std::cout << '\n';
    }
    if (r.receive) {
        std::cout << "  probe=" << r[ProbeMisses] << '/' << r[ProbeCalls]
                  << '/' << r[ProbeMissTicks] << '/' << r[ProbeFlagBlocks] << '/' << r[ProbeConfirmedSlots];
        if (r.present[ProbeBlockedSlot] && r[ProbeMisses]) {
            std::cout << " blocked=t" << r[ProbeBlockedToken] << "/k";
            if (r[ProbeBlockedSlot] == 0xffffffffU) std::cout << "all";
            else std::cout << r[ProbeBlockedSlot];
        }
        std::cout << " inspect=" << r[TailInspectTicks];
        if (r[TailSampleStride])
            std::cout << " sampled=" << r[TailSampleStride] << '/' << r[TailInspectCalls] << '/' << r[TailInspectSkipped];
        std::cout << '\n';
    }
    if (r[EarlyIssueTicks] || r[MetadataBatches] || r[PollPauseCalls])
        std::cout << "  extra early/meta/pause_calls/pause_ticks=" << r[EarlyIssueTicks] << '/' << r[MetadataBatches]
                  << '/' << r[PollPauseCalls] << '/' << r[PollPauseTicks] << '\n';
    if (r.receive && (r[WqePreparedPackets] || r[WqeFallbackPackets] || r[MtePackPackets]))
        std::cout << "  xfer=" << r[WqePrepareTicks] << '/'
                  << r[WqePreparedPackets] << '/' << r[WqeFallbackPackets] << '/' << r[MtePackPackets] << '\n';
    if (r.receive && r[PartialInputLanes])
        std::cout << "  input=" << r[PartialInputLanes] << '/' << r[ReadyInputIssued]
                  << '/' << r[ReadyInputUsed] << " vec=" << r[PartialPairGroups]
                  << '/' << r[PartialVectorBarriers] << " arith=" << r[PartialStreamSteps]
                  << '/' << r[PartialFrontLoads] << '\n';
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
        if (r[RouteFilterRows] || r[RouteCompactTicks])
            std::cout << " compact=" << r[RouteFirstHopRows] << '/' << r[RouteFilterRows] << '/' << r[RouteCompactTicks];
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
// Same real collective iteration, never a concatenation of unrelated samples.
// These are independent maxima with core-local origins, not additive durations.
void compact_supply(const std::vector<Record>& records)
{
    const Record *prep=nullptr,*drain=nullptr,*notify=nullptr,*group=nullptr,*input=nullptr;
    auto interval=[](const Record& r,Word begin,Word end) {
        return r.present[begin] && r.present[end] && r[end]>=r[begin] ? r[end]-r[begin] : 0U;
    };
    for (const auto& r:records) {
        if(r.present[StageEnd] && (!prep || r[StageEnd]>(*prep)[StageEnd])) prep=&r;
        if(r.present[StageDrained] && (!drain || r[StageDrained]>(*drain)[StageDrained])) drain=&r;
        if(r.present[NotifyEnd] && (!notify || interval(r,NotifyBegin,NotifyEnd)>interval(*notify,NotifyBegin,NotifyEnd))) notify=&r;
        if(r.present[ReadyObserved] && (!group || interval(r,ReadyBegin,ReadyObserved)>interval(*group,ReadyBegin,ReadyObserved))) group=&r;
        if(r[Level]>=2 && r.receive && (!input || r[PartialLocalWaitTicks]+r[PartialGetWaitTicks]>
            (*input)[PartialLocalWaitTicks]+(*input)[PartialGetWaitTicks])) input=&r;
    }
    std::cout << " X";
    if(prep) std::cout << " prep=" << (*prep)[Rank] << '/' << (*prep)[Core] << ':' << (*prep)[StageEnd];
    if(drain) std::cout << " drained=" << (*drain)[Rank] << '/' << (*drain)[Core] << ':' << (*drain)[StageDrained];
    if(notify) std::cout << " notify=" << (*notify)[Rank] << '/' << (*notify)[Core] << ':'
        << interval(*notify,NotifyBegin,NotifyLocalReady) << '/' << interval(*notify,NotifyLocalReady,NotifyEnd);
    if(group) std::cout << " group=" << (*group)[Rank] << '/' << (*group)[Core] << ':'
        << ((*group)[ReadyTaskRounds]?"O":"W") << interval(*group,ReadyBegin,ReadyObserved)
        << '/' << (*group)[ReadyMissingRank] << '/' << (*group)[ReadyMissingCore] << '/' << (*group)[ReadyMissingCount];
    if(input) std::cout << " inputwait=" << (*input)[Rank] << '/' << (*input)[Core] << ':'
        << (*input)[PartialLocalWaitTicks] << '/' << (*input)[PartialGetWaitTicks];
    std::cout << '\n';
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
    std::cout << "LL_COMBINE compact=v3 global rank-MAX; Event=us; profile=ticks/core-entry; no samples removed\n"
                 "D/IQR/Tfreq use the phase number from the variant table.\n"
                 "p50=full counters; worst=R timeline/RDMA/tail, S send_end=last_submit/direct/pipe; full CSV/Perfetto.\n"
                 "IQR=candidates; WIDE=(max-min)/P50>20%; full samples/fields in CSV; '-'=absent.\n"
                 "e(S)=init/stage/core_notify/notify/direct_first/direct_last/direct_end/pipe\n"
                 "e(R)=init/stage_drained/core_notify/ready_begin/ready_observed/ready_end/partial_end/put_first/put_last/final_first/final_last/store/drain/pipe\n"
                 "cache=mask(stage1/TX2/result4); m3=input/store_issue/drain; g=load/check/calls; m5=input_issue/input_wait/compute/store; ticks, NOT cache hits\n"
                 "m4=local_issue/get_issue/local_wait/get_wait/compute_issue/output_wait/pack/input_drain cumulative ticks\n"
                 "tiny=policy/probes/confirmed_slots/skipped_flags (policy bits: setup=1,sparse=2); small=output_init_lanes/setup_ticks/meta_wait_ticks/single_task; direct_meta=cached/reloaded/active_lanes; route_plan=preload_rows/wait_ticks/gather_rows\n"
                 "X=same-sample global maxima r/c:ticks; prep/drained=own-entry offsets, notify=local_wait/publish, group=W(wait)|O(observe)/missing_rank/core/count, inputwait=local/get; not additive\n"
                 "top4 cumulative ticks: s.=stage_, d.=direct_, m4.=partial_, m5.=final_, g.=ready_; overlapping costs are not additive.\n"
                 "rdma=n/deferred/gap/stage_issue/stage_wait/put/quiet; stage=wait_calls/pairs/prefetch_calls; stream=packed_posts/eager/ticks\n"
                 "txwait=prefix_remaining/copy_remaining/calls (profile-only event split, not pure DMA latency); priority=local_rows_deferred/extra_scan_ticks\n"
                 "lg=local/get; xfer=prepare_ticks/prepared/fallback/mte_pack; input=lanes/early_reads/used_reads; vec=pair_groups/barriers; arith=stream_steps/front_loads\n"
                 "probe=miss/calls/miss_ticks/flags/confirmed; N=r/c local_wait/publish miss/calls/max_probe missing_core/count stores inspect\n"
                 "P=own-entry rank/core:first_put/last_put/partial_end/packets\n"
                 "g=miss/calls/max_probe/inspect missing_rank/core/count; 4294967295=all\n"
                 "G=r/c window=W(wait)|O(observe, includes work) miss/calls/max_probe missing_rank/core/count inspect; omitted if same as R\n"
                 "q=first/all/done rounds/stalls/skips before_all/remote cached/reloads setup/plan/filter/snapshot/pipeline_issue retire\n"
                 "scan=GM_batches/rows/Vector_group_passes (16B/row); ub=snapshot_bytes/capacity/fallbacks; compact=FIRST_HOP/filter_rows/ticks\n"
                 "T=last observed miss (schema26: first + every8 repeated misses/token); time=first/last/ready; src=slot:rank/core/path/type; D=direct P=partial.\n"
                 "O=upstream role; active=scatter/gather/RDMA/final cores per rank; policy=0 reference/1 retire/2 overlap; retire=start/end.\n"
                 "oe=init/scatter/first_ready/gather/put_first/put_last/sender/barrier_begin/barrier_end/final_init/first/last/store/end; ops poll=miss/calls/ticks, NOT wire latency.\n"
                 "O reduce_store=gather reduction/output/final reduction/output ticks (includes input loads or output packing).\n"
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
            if (d.config.count("upstream_revision") && !d.config.at("upstream_revision").empty())
                std::cout << "REF " << diagnostic_id(d.variant) << " upstream=" << d.config.at("upstream_revision")
                          << " transport=SHMEM-per-core-QP; policy="
                          << (d.config.count("ops_policy") ? d.config.at("ops_policy") : "0") << "; "
                          << (d.records.empty() ? "no internal profile" : "instrumented diagnostics") << '\n';
            std::vector<double> values;
            for (const auto& [i, e] : d.events) values.push_back(e.us);
            std::sort(values.begin(), values.end());
            const double q1 = percentile(values, .25), q3 = percentile(values, .75), limit = q3 + 1.5 * (q3 - q1);
            std::vector<std::pair<uint32_t, Event>> flagged;
            for (const auto& [i, e] : d.events) if (e.us > limit) flagged.push_back({i, e});
            if (!flagged.empty()) {
                std::cout << "IQR " << diagnostic_id(d.variant) << " n=" << flagged.size() << " limit=" << limit << " i/r/us=";
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
                std::cout << "D " << diagnostic_id(d.variant) << ' ' << (i == worst ? "worst" : "p50") << " i=" << i
                          << " event_r=" << event.rank << " us=" << event.us << '\n';
                if (!records.empty() && records.front()[OpsRole]) {
                    std::set<std::pair<uint64_t, uint64_t>> printed;
                    // Real records from the Event rank: longest active scatter,
                    // gather, sender and final worker; never synthesize a pipeline.
                    for (uint32_t role = 1; role <= 4; ++role) {
                        const Record* selected = nullptr;
                        const Word end = role == 1 ? StageEnd : role == 2 ? PartialEnd : role == 3 ? DirectEnd : FinalEnd;
                        for (const auto& r : records)
                            if (r[Rank] == event.rank && r.present[end] &&
                                (role == 4 ? r[FinalTokens] != 0 : r[OpsRole] == role) &&
                                (!selected || r[end] > (*selected)[end])) selected = &r;
                        if (selected && printed.emplace((*selected)[Rank], (*selected)[Core]).second) compact_record(*selected);
                    }
                    const Record* lastPut = nullptr;
                    for (const auto& r : records) if (r.present[RdmaLastPut] &&
                        (!lastPut || r[RdmaLastPut] > (*lastPut)[RdmaLastPut])) lastPut = &r;
                    if (lastPut && printed.emplace((*lastPut)[Rank], (*lastPut)[Core]).second) {
                        std::cout << " O_latest_put_offset (independent core origins)\n";
                        compact_record(*lastPut);
                    }
                    continue;
                }
                compact_supply(records);
                const Record* receiveRecord = nullptr;
                for (bool receive : {false, true}) {
                    const Record* tail = nullptr;
                    for (const auto& r : records)
                        if (!r[CoalesceWorker] && r[Rank] == event.rank && r.receive == receive &&
                            (!tail || r[FlushDone] > (*tail)[FlushDone])) tail = &r;
                    if (!tail) throw std::runtime_error("Missing Event rank diagnostic role");
                    compact_record(*tail, i == worst && i != median);
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
