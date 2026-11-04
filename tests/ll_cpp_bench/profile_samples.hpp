#pragma once
#include "profile.hpp"
#include "mpi_coordinator.hpp"
#include "options.hpp"
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>

namespace llbench {
// Four selections per iteration: longest send/receive body, latest local
// Put-return offset. Full per-core records remain in each rank's CSV.
// Also keep the slowest coalescing worker; duplicates are removed on export.
class ProfileSamples {
    std::vector<uint64_t> records_;
public:
    void capture(const std::vector<uint64_t>& data, uint32_t cores, uint32_t recv)
    {
        using namespace CombineProfile;
        if (!recv || recv >= cores || data.size() != size_t(cores) * kWords)
            throw std::runtime_error("Invalid profile sample dimensions");
        uint32_t selected[4] = {0, cores - recv, 0, 0};
        for (uint32_t c=0;c<cores-recv;++c)
            if (!data[c*kWords+CoalesceWorker]) { selected[0]=c; break; }
        for (uint32_t c = 0; c < cores; ++c) {
            const auto* row = data.data() + c * kWords;
            const auto role = c < cores - recv ? 0U : 1U;
            if (!row[CoalesceWorker] && profile_interval(row, Entry, FlushDone) >
                profile_interval(data.data() + selected[role] * kWords, Entry, FlushDone)) selected[role] = c;
            if (profile_interval(row, Entry, RdmaLastPut) >
                profile_interval(data.data() + selected[2] * kWords, Entry, RdmaLastPut)) selected[2] = c;
            const auto* worker=data.data()+selected[3]*kWords;
            if (row[CoalesceWorker] && (!worker[CoalesceWorker] ||
                profile_interval(row,Entry,FlushDone)>profile_interval(worker,Entry,FlushDone))) selected[3]=c;
        }
        if (!data[selected[3]*kWords+CoalesceWorker]) selected[3]=selected[0];
        for (auto c : selected) records_.insert(records_.end(), data.begin() + c * kWords, data.begin() + (c + 1) * kWords);
    }
    void write(const Coordinator& peers, const Options& o, uint32_t bs, uint32_t cores, uint32_t recv,
               const std::vector<float>& times)
    {
        using namespace CombineProfile;
        if (times.empty() || records_.size() != times.size() * 4 * kWords)
            throw std::runtime_error("Missing profile samples");
        std::ofstream out;
        if (!o.rank) {
            const auto path = std::filesystem::path(o.output) / "diagnostic.csv";
            const bool exists = std::filesystem::exists(path);
            out.open(path, std::ios::app);
            if (!exists) profile_header(out);
        }
        for (uint32_t i = 0; i < times.size(); ++i) {
            std::vector<uint64_t> send(1 + 4 * kWords);
            send[0] = o.device;
            std::copy_n(records_.data() + size_t(i) * 4 * kWords, 4 * kWords, send.data() + 1);
            // A few KiB per rank, after sampling; no communication in the timed path.
            const auto all = peers.gather(send.data(), send.size() * sizeof(uint64_t));
            if (o.rank) continue;
            for (uint32_t rank = 0; rank < o.world; ++rank) {
                std::memcpy(send.data(), all.data() + rank * send.size() * sizeof(uint64_t), send.size() * sizeof(uint64_t));
                std::vector<uint64_t> data(size_t(cores) * kWords);
                std::set<uint32_t> selected;
                for (uint32_t j = 0; j < 4; ++j) {
                    auto row = send.data() + 1 + j * kWords;
                    if (row[Core] >= cores) throw std::runtime_error("Invalid selected core");
                    selected.insert(uint32_t(row[Core]));
                    std::copy_n(row, kWords, data.data() + row[Core] * kWords);
                }
                profile_rows(out, data, cores, recv, rank, o.world, bs, uint32_t(send[0]), o.profile, i,
                             {selected.begin(), selected.end()});
            }
        }
        if (!o.rank) {
            out.close();
            if (!out) throw std::runtime_error("Cannot write global diagnostic records");
        }
    }
};
}  // namespace llbench
