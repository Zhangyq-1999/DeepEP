#pragma once
#include "profile.hpp"
#include "mpi_coordinator.hpp"
#include "options.hpp"
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>

namespace llbench {
// Three records per iteration: longest send/receive body and latest local
// Put-return offset. Full per-core records remain in each rank's CSV.
class ProfileSamples {
    std::vector<uint64_t> records_;
public:
    void capture(const std::vector<uint64_t>& data, uint32_t cores, uint32_t recv)
    {
        using namespace CombineProfile;
        if (!recv || recv >= cores || data.size() != size_t(cores) * kWords)
            throw std::runtime_error("Invalid profile sample dimensions");
        uint32_t selected[3] = {0, cores - recv, 0};
        for (uint32_t c = 0; c < cores; ++c) {
            const auto* row = data.data() + c * kWords;
            const auto role = c < cores - recv ? 0U : 1U;
            if (profile_interval(row, Entry, FlushDone) >
                profile_interval(data.data() + selected[role] * kWords, Entry, FlushDone)) selected[role] = c;
            if (profile_interval(row, Entry, RdmaLastPut) >
                profile_interval(data.data() + selected[2] * kWords, Entry, RdmaLastPut)) selected[2] = c;
        }
        for (auto c : selected) records_.insert(records_.end(), data.begin() + c * kWords, data.begin() + (c + 1) * kWords);
    }
    void write(const Coordinator& peers, const Options& o, uint32_t bs, uint32_t cores, uint32_t recv,
               const std::vector<float>& times)
    {
        using namespace CombineProfile;
        if (times.empty() || records_.size() != times.size() * 3 * kWords)
            throw std::runtime_error("Missing profile samples");
        std::vector<uint32_t> order(times.size());
        std::iota(order.begin(), order.end(), 0U);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return times[a] < times[b]; });
        const std::set<uint32_t> iterations{order[(order.size() - 1) / 2], order.back()};
        std::ofstream out;
        if (!o.rank) {
            const auto path = std::filesystem::path(o.output) / "diagnostic.csv";
            const bool exists = std::filesystem::exists(path);
            out.open(path, std::ios::app);
            if (!exists) profile_header(out);
        }
        for (auto i : iterations) {
            std::vector<uint64_t> send(1 + 3 * kWords);
            send[0] = o.device;
            std::copy_n(records_.data() + size_t(i) * 3 * kWords, 3 * kWords, send.data() + 1);
            // A few KiB per rank, after sampling; no communication in the timed path.
            const auto all = peers.gather(send.data(), send.size() * sizeof(uint64_t));
            if (o.rank) continue;
            for (uint32_t rank = 0; rank < o.world; ++rank) {
                std::memcpy(send.data(), all.data() + rank * send.size() * sizeof(uint64_t), send.size() * sizeof(uint64_t));
                std::vector<uint64_t> data(size_t(cores) * kWords);
                std::set<uint32_t> selected;
                for (uint32_t j = 0; j < 3; ++j) {
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
