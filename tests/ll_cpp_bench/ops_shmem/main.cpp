#include "fixture.hpp"
#include "../sample_report.hpp"
#include "../profile_samples.hpp"
#ifndef LL_BENCH_PYTHON
#include "../mpi_coordinator.hpp"
#endif
#include "../variant_api.hpp"
#include "../entry_sync.hpp"
#include "runtime/ll_core_config.hpp"
#include "host/mem/shmem_host_heap.h"
#include "host/data_plane/shmem_host_rma.h"
#include "host/data_plane/shmem_host_p2p_sync.h"
#include "host/data_plane/shmem_host_cc.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

namespace {
void check(int rc, const char* operation)
{
    if (rc) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}
struct Buffer {
    uint8_t* p = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(std::max(n, size_t(32)))
    {
        check(aclrtMalloc(reinterpret_cast<void**>(&p), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        check(aclrtMemset(p, bytes, 0, bytes), "clear allocation");
    }
    ~Buffer()
    {
        if (p) aclrtFree(p);
    }
    Buffer(const Buffer&) = delete;
    void upload(const void* data, size_t n)
    {
        if (n) check(aclrtMemcpy(p, bytes, data, n, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
    }
    void download(void* data, size_t n)
    {
        if (n) check(aclrtMemcpy(data, n, p, n, ACL_MEMCPY_DEVICE_TO_HOST), "download");
    }
};
struct Event {
    aclrtEvent value = nullptr;
    Event()
    {
        check(aclrtCreateEventWithFlag(&value, ACL_EVENT_TIME_LINE), "create Event");
    }
    ~Event()
    {
        if (value) aclrtDestroyEvent(value);
    }
    void record(aclrtStream stream)
    {
        check(aclrtRecordEvent(value, stream), "record Event");
    }
};
struct DeviceCase {
    uint32_t bs;
    opsbench::Fixture data;
    opsbench::Args args{};
    std::vector<std::unique_ptr<Buffer>> buffers;
    Buffer& allocate(size_t n)
    {
        buffers.emplace_back(std::make_unique<Buffer>(n));
        return *buffers.back();
    }
    template <class T>
    uint8_t* upload(const std::vector<T>& v)
    {
        auto& b = allocate(v.size() * sizeof(T));
        b.upload(v.data(), v.size() * sizeof(T));
        return b.p;
    }
    DeviceCase(const llbench::Options& o, uint32_t bs, uint32_t cores, uint64_t ub, uint8_t* context) : bs(bs), data(o, bs)
    {
        args.context = context;
        args.cores = cores;
        args.bf16 = o.dtype == "bf16";
        args.expanded = upload(data.expanded);
        args.experts = upload(data.experts);
        args.assist = upload(data.assist);
        args.counts = upload(data.counts);
        args.weights = upload(data.weights);
        args.expandedWeights = upload(data.expandedWeights);
        args.output = allocate(data.expected.size() * 2).p;
        opsbench::Config c{o.rank,    o.world,   bs,    o.hidden, o.topk,
                           o.experts, data.rows, cores, ub,       shmemx_get_ffts_config()};
        auto& config = allocate(sizeof(c));
        config.upload(&c, sizeof(c));
        args.config = config.p;
    }
    double precision(bool bf16)
    {
        std::vector<uint16_t> actual(data.expected.size());
        check(aclrtMemcpy(actual.data(), actual.size() * 2, args.output, actual.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST),
              "read output");
        double worst = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double expected = llbench::decode(data.expected[i], bf16), value = llbench::decode(actual[i], bf16);
            const double error = std::abs(value - expected);
            worst = std::max(worst, error);
            if (!std::isfinite(value) || error > (bf16 ? .02 : .002) * (1 + std::abs(expected)))
                throw std::runtime_error("ops-shmem-base precision failed at element " + std::to_string(i));
        }
        return worst;
    }
    void flipExpertSign(const llbench::Options&)
    {
        // Exercise new payloads at the same addresses across upstream epochs.
        for (auto& value : data.expanded) value ^= 0x8000U;
        for (auto& value : data.expected) value ^= 0x8000U;
        if (!data.expanded.empty())
            check(aclrtMemcpy(args.expanded, data.expanded.size() * 2, data.expanded.data(),
                              data.expanded.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "replace ops inputs");
    }
};
opsbench::Context context_for(const llbench::Runtime& session)
{
    opsbench::Context c{};
    c.rank = session.options.rank;
    c.ranksPerServer = std::min(session.options.world, session.options.ranksPerHost);
    const uint32_t server = c.rank / c.ranksPerServer, servers = session.options.world / c.ranksPerServer;
    for (uint32_t peer = server * c.ranksPerServer; peer < (server + 1) * c.ranksPerServer; ++peer) {
        void* address = peer == c.rank ? session.workspace : aclshmem_ptr(session.workspace, peer);
        if (!address) throw std::runtime_error("Missing SHMEM local mapping");
        c.localWindows[peer] = reinterpret_cast<uint64_t>(address);
    }
    for (uint32_t s = 0, channel = 0; s < servers; ++s) {
        // Local symmetric addresses are intentional: SHMEM translates the target.
        c.remoteWindows[s] = reinterpret_cast<uint64_t>(session.workspace) + opsbench::windowBytes;
        if (s != server) {
            c.peers[channel] = s * c.ranksPerServer + c.rank % c.ranksPerServer;
            channel += 2;  // Preserve the upstream context's two entries per server.
        }
    }
    c.remoteWindowBytes = opsbench::windowBytes;
    return c;
}
}  // namespace

#ifdef LL_BENCH_PYTHON
// Reuse the same Python session, launch gate and Event boundary as DeepEP.
using Case = DeviceCase;
#include "../../ll_python_bench/bindings.hpp"
#else
LL_BENCH_EXPORT void ll_bench_run(const llbench::Runtime* shared)
{
    static_assert(llbench::workspaceBytes == opsbench::workspaceBytes);
    const auto& session = *shared;
    const auto& peers = *shared->peers;
    auto o = shared->options;
    const auto stream = shared->stream;
    const auto cores = shared->cores;
    if (!DEEPEP_BENCH_PROFILE) o.profile = 0;
    if (o.profile) o.iterations = o.profileIterations;
    const auto recv = cores / 2U;
    if ((o.hidden != 3584 && o.hidden != 7168) || o.topk != 8 || o.experts != 384)
        throw std::runtime_error("Initial ops-shmem baseline supports H3584/H7168, TopK8, E384");
    uint64_t ub = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub < 128 * 1024) throw std::runtime_error("Cannot query sufficient UB capacity");
    peers.verify(llbench::hash(o.signature() + "/ops-shmem/" + DEEPEP_BENCH_SOURCE_ID + "/" + std::to_string(cores) +
                               "/" + std::to_string(ub)));
    std::filesystem::create_directories(o.output);
    const auto directory = std::filesystem::path(o.output);
    const auto rankPath = directory / ("rank-" + std::to_string(o.rank) + ".csv");
    if (std::filesystem::exists(rankPath)) throw std::runtime_error("Result exists; choose a fresh run-id");
    std::ofstream samples(rankPath), summary, profileCsv;
    if (!samples) throw std::runtime_error("Cannot open samples");
    samples << "tokens,iteration,combine_event_us,implementation,entry_sync,profile_build,profile_level\n";
    if (o.profile) {
        profileCsv.open(directory / ("rank-" + std::to_string(o.rank) + "-profile.csv"));
        if (!profileCsv) throw std::runtime_error("Cannot open profile CSV");
        llbench::profile_header(profileCsv);
    }
    if (o.rank == 0) {
        summary.open(directory / "summary.csv");
        if (!summary) throw std::runtime_error("Cannot open summary");
        summary << "ep,tokens,hidden,topk,experts,dtype,route,seed,capacity,warmup,samples,active_aiv,implementation,"
                   "source_id,upstream_revision,entry_sync,profile_build,profile_level,rank_max_mean_us,rank_max_min_us,rank_max_max_us,p50_us,p99_"
                   "us\n";
    }
    auto c = context_for(session);
    Buffer context(sizeof(c));
    context.upload(&c, sizeof(c));
    if (!o.rank)
        std::cout << "ops-shmem-base: upstream " << LL_OPS_REV
                  << "; SHMEM QP Put + batch-boundary quiet; entry_sync=" << o.entrySync << std::endl;
    llbench::EntrySync entry(session);
    if (o.entrySync == "device") {
        session.report("device entry sync preparation");
        entry.enqueue([] {});
        entry.complete("device entry sync preparation");
    }
    for (auto bs : o.tokens) {
        session.report("ops input preparation", bs);
        DeviceCase item(o, bs, cores, ub, context.p);
        auto launch = [&] { ops_shmem_combine_do(stream, &item.args); };
        session.report("Combine correctness", bs);
        entry.enqueue(launch);
        entry.complete("Combine correctness", bs);
        double error = item.precision(o.dtype == "bf16");
        peers.barrier();
        for (uint32_t generation = 0; generation < 4; ++generation) {
            if (generation % 2 == 0) item.flipExpertSign(o);
            peers.barrier();
            entry.enqueue(launch);
            entry.complete("cache reuse correctness", bs, generation);
            error = std::max(error, item.precision(o.dtype == "bf16"));
            peers.barrier();
        }
        Event start, end;
        std::vector<uint64_t> profileData;
        std::unique_ptr<Buffer> profileBuffer;
        llbench::ProfileSamples profiles;
        if (o.profile) {
            profileData.resize(size_t(cores) * CombineProfile::kWords);
            profileBuffer = std::make_unique<Buffer>(profileData.size() * sizeof(uint64_t));
            item.args.stageTimer = profileBuffer->p;
            item.args.timerMode = o.profile;
        }
        std::vector<float> times;
        for (uint32_t i = 0; i < o.warmup + o.iterations; ++i) {
            const bool sampling = i >= o.warmup;
            const auto iteration = sampling ? i - o.warmup : i;
            if (iteration % 10U == 0)
                session.report(sampling ? "sampling" : "warmup", bs, iteration, sampling ? o.iterations : o.warmup);
            if (o.profile) {
                std::fill(profileData.begin(), profileData.end(), 0U);
                profileBuffer->upload(profileData.data(), profileBuffer->bytes);
            }
            entry.enqueue([&] {
                if (sampling) start.record(stream);
                launch();
                if (sampling) end.record(stream);
            });
            entry.complete(sampling ? "sampling" : "warmup", bs, iteration);
            if (i >= o.warmup) {
                float ms = 0;
                check(aclrtEventElapsedTime(&ms, start.value, end.value), "elapsed Event");
                if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid Event duration");
                times.push_back(ms * 1000);
                samples << bs << ',' << i - o.warmup << ',' << times.back() << ",ops-shmem-base," << o.entrySync
                        << ',' << DEEPEP_BENCH_PROFILE << ',' << o.profile << '\n';
                if (o.profile) {
                    profileBuffer->download(profileData.data(), profileBuffer->bytes);
                    llbench::profile_rows(profileCsv, profileData, cores, recv, o.rank, o.world, bs, o.device,
                                          o.profile, iteration);
                    profiles.capture(profileData, cores, recv, o.profileFrequencyHz != 0);
                }
            }
        }
        session.report("precision check and result gather", bs, o.iterations, o.iterations);
        item.precision(o.dtype == "bf16");
        samples.flush();
        auto gathered = peers.gather(times.data(), times.size() * sizeof(float));
        const auto rankMax = llbench::rank_max_samples(gathered, o, bs);
        if (o.profile) {
            profileCsv.flush();
            profiles.write(peers, o, bs, cores, recv, rankMax, DEEPEP_BENCH_SOURCE_ID);
        }
        if (!o.rank) {
            auto s = llbench::stats(rankMax);
            summary << o.world << ',' << bs << ',' << o.hidden << ',' << o.topk << ',' << o.experts << ',' << o.dtype
                    << ',' << o.route << ',' << o.seed << ',' << o.capacity << ',' << o.warmup << ',' << o.iterations
                    << ',' << cores << ",ops-shmem-base," << DEEPEP_BENCH_SOURCE_ID << ',' << LL_OPS_REV << ','
                    << o.entrySync << ',' << DEEPEP_BENCH_PROFILE << ',' << o.profile
                    << ',' << s.mean << ',' << s.min << ',' << s.max << ',' << s.p50 << ',' << s.p99
                    << '\n';
            summary.flush();
            std::cout << "BS=" << bs << " rank-MAX Event us: mean=" << s.mean << " range=[" << s.min << ',' << s.max
                      << "] p99=" << s.p99 << " rank0_max_abs_error=" << error << std::endl;
        }
        peers.barrier();
    }
}
#endif
