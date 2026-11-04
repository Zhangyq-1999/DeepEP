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
    std::cout << " T last-miss t=" << r[TailToken] << " first=k" << r[TailSlot] << "/b" << r[TailBlock]
              << " missing_blocks=" << r[TailMissingBlocks] << " seen=" << r[TailObservations]
              << " first/last/ready=" << r[TailFirstSeen] << '/' << r[TailLastSeen] << '/' << r[TailReady]
              << " producers=";
    for (uint32_t slot = 0; slot < 32; ++slot) if ((r[TailMissingMask] >> slot) & 1U) {
        const auto found = d.producers.find({uint32_t(r[Rank]), uint32_t(r[TailToken]), slot});
        if (found == d.producers.end()) { std::cout << "k" << slot << ":unknown "; continue; }
        const auto& p = found->second;
        std::cout << 'k' << slot << ":r" << p.rank << "/c" << p.core << '/' << p.path
                  << '/' << (p.type == 1 ? "DIRECT" : "PARTIAL") << ' ';
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
              << " producer>receiver/path/type:n=";
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
    std::cout << "  end";
    for (Word w : r.receive ? std::vector<Word>{InitEnd, ReadyEnd, PartialEnd, RdmaFirstPut, RdmaLastPut, FinalFirstReady, FinalLastReady, FinalLastStore, DrainEnd, PipeDone}
                           : std::vector<Word>{InitEnd, StageEnd, NotifyEnd, DirectFirstSubmit, DirectLastSubmit, DirectEnd, PipeDone})
        if (r.present[w]) std::cout << ' ' << names[w] << '=' << r[w];
    std::cout << "\n  top";
    const auto costs = top_costs(r, kDetailBegin, kDetailEnd);
    for (size_t j = 0; j < std::min<size_t>(6, costs.size()); ++j)
        std::cout << ' ' << names[costs[j]] << '=' << r[costs[j]];
    std::cout << " poll=miss/call/ticks:" << r[ProbeMisses] << '/' << r[ProbeCalls] << '/' << r[ProbeMissTicks]
              << " flags/confirmed=" << r[ProbeFlagBlocks] << '/' << r[ProbeConfirmedSlots];
    if (r.present[ProbeBlockedSlot] && r[ProbeMisses]) {
        std::cout << " blocked=t" << r[ProbeBlockedToken] << "/k";
        if (r[ProbeBlockedSlot] == 0xffffffffU) std::cout << "all";
        else std::cout << r[ProbeBlockedSlot];
    }
    std::cout << " early_issue=" << r[EarlyIssueTicks]
              << " meta_batches=" << r[MetadataBatches] << " tail_inspect=" << r[TailInspectTicks]
              << " pause=calls/ticks:" << r[PollPauseCalls] << '/' << r[PollPauseTicks]
              << " packets/defer/gap=" << r[RdmaPackets] << '/' << r[RdmaDeferredPackets] << '/' << r[RdmaDeferGapTicks] << '\n';
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
    std::cout << "LL_COMBINE global rank-MAX; Event=us; profile=ticks/core-entry; no samples removed\n"
                 "IQR flags are candidates; WIDE means (max-min)/P50 > 20%. Detailed CSVs retained.\n";
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
            std::cout << std::left << std::setw(24) << d.variant << std::setw(8) << d.kind << std::right
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
            std::cout << "E " << d.variant << " i/r/us=";
            for (const auto& [i, e] : d.events) std::cout << i << '/' << e.rank << '/' << e.us << ' ';
            std::cout << '\n';
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
                for (bool receive : {false, true}) {
                    const Record* tail = nullptr;
                    for (const auto& r : records)
                        if (!r[CoalesceWorker] && r[Rank] == event.rank && r.receive == receive &&
                            (!tail || r[FlushDone] > (*tail)[FlushDone])) tail = &r;
                    if (!tail) throw std::runtime_error("Missing Event rank diagnostic role");
                    compact_record(*tail);
                    if (receive) tail_producers(d, *tail);
                }
                const Record* worker=nullptr;
                for(const auto& r:records) if(r[Rank]==event.rank && r[CoalesceWorker] &&
                    (!worker || r[FlushDone]>(*worker)[FlushDone])) worker=&r;
                if(worker) compact_record(*worker);
                std::vector<const Record*> puts;
                for (const auto& r : records) if (r.present[RdmaLastPut]) puts.push_back(&r);
                std::stable_sort(puts.begin(), puts.end(), [](auto a, auto b) { return (*a)[RdmaLastPut] > (*b)[RdmaLastPut]; });
                std::cout << " P own-entry r/c:first_put/last_put/partial_end/packets=";
                for (size_t j = 0; j < std::min<size_t>(2, puts.size()); ++j) {
                    const auto& r = *puts[j];
                    std::cout << r[Rank] << '/' << r[Core] << ':' << r[RdmaFirstPut] << '/' << r[RdmaLastPut]
                              << '/' << r[PartialEnd] << '/' << r[RdmaPackets] << ' ';
                }
                std::cout << '\n';
            }
        }
    }
}
