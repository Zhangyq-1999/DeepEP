#include <cstdint>
#define GM_ADDR uint8_t*
#include "transport.hpp"
#include <cassert>
#include <iostream>

int main()
{
    uint8_t local[65536]{}, remote[65536]{}, scratch[512]{};
    static_assert(opsbench::smallShape(24, 64) && !opsbench::smallShape(25, 64) &&
                  !opsbench::smallShape(1, 128), "experiment boundaries");
    Model::local = local;
    Model::remote = remote;
    AscendC::core = 53;
    std::fill(local + 512, local + 1024, 0x21);
    std::fill(local + 1024, local + 1536, 0x43);
    opsbench::Transport transport;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    std::vector<uint64_t> records(56 * CombineProfile::kWords);
    CombineProfile::DeviceProfile profile;
    profile.Init(reinterpret_cast<uint8_t*>(records.data()), 2);
    transport.SetProfile(&profile);
#endif
    transport.Init({scratch}, sizeof(scratch));
#if DEEPEP_LL_OPS_POLICY
    transport.EnableDeferred(true);
#endif
    auto batch = transport.MakeBatchHandle(17, {scratch}, sizeof(scratch), local, local);
    transport.WriteNbi(batch, local + 4096, local + 512, 512);
    transport.WriteNbi(batch, local + 4608, local + 1024, 512);
    assert(Model::pending.size() == 2 && Model::quiets == 0);
    for (const auto& request : Model::pending) {
        assert(request.peer == 17 && request.qp == 53);
        assert(request.src >= local + 512 && request.src < local + 1536);
        assert(request.dst >= remote + 4096 && request.dst < remote + 5120);
    }
    transport.BatchCommit(batch);
#if DEEPEP_LL_OPS_POLICY
    assert(Model::pending.size() == 2 && Model::quiets == 0 && transport.HasPending());
    transport.BatchCommit(batch);  // empty commit must not lose pending peers
    auto second = transport.MakeBatchHandle(33, {scratch}, sizeof(scratch), local, local);
    transport.WriteNbi(second, local + 5120, local + 512, 512);
    transport.BatchCommit(second);
    // Destroy the old UB view and provide fresh scratch: GM sources survive.
    std::memset(scratch, 0xcc, sizeof(scratch));
    uint8_t retiredScratch[128]{};
    transport.Retire({retiredScratch});
    assert(Model::pending.empty() && Model::quiets == 2 && !transport.HasPending());
    assert(std::memcmp(remote + 5120, local + 512, 512) == 0);
    transport.Retire({retiredScratch});
    assert(Model::quiets == 2);
    transport.EnableDeferred(false);  // >24 BS fallback preserves batch waits
    transport.WriteNbi(batch, local + 4096, local + 512, 512);
    transport.BatchCommit(batch);
    assert(Model::pending.empty() && Model::quiets == 3);
#else
    assert(Model::pending.empty() && Model::quiets == 1);
#endif
    assert(std::memcmp(remote + 4096, local + 512, 1024) == 0);
    transport.BatchCommit(batch);
    assert(Model::quiets == (opsbench::policy ? 3U : 1U));
#ifdef DEEPEP_LL_COMBINE_PROFILE
    profile.Mark(CombineProfile::FlushDone);
    profile.Finish(1, 32, 2);
    using namespace CombineProfile;
    const auto* row = records.data() + 53 * kWords;
    assert(row[RdmaPackets] == (opsbench::policy ? 4U : 2U));
    assert(row[RdmaBytes] == (opsbench::policy ? 2048U : 1024U));
    assert(row[RdmaQuietCalls] == (opsbench::policy ? 3U : 1U));
    assert(row[OpsDeferredCommits] == (opsbench::policy ? 2U : 0U));
    assert(row[OpsRetiredPeers] == (opsbench::policy ? 2U : 0U));
    assert((row[Valid] & (1ULL << RdmaFirstPut)) && (row[Valid] & (1ULL << RdmaLastPut)));
#endif
#if DEEPEP_LL_OPS_POLICY
    // 64P/4-host example: three remote same-slot ranks, 24 immutable payloads
    // each. Commit after every ready row (worst fragmented availability), but
    // retire only once per peer. New epochs reuse addresses with changed data.
    for (unsigned generation = 1; generation <= 6; ++generation) {
        transport.Init({scratch}, sizeof(scratch));
        transport.EnableDeferred(true);
        const auto quietBefore = Model::quiets;
        for (unsigned server = 0; server < 3; ++server) {
            auto outgoing = transport.MakeBatchHandle(31 + server * 16, {scratch}, sizeof(scratch), local, local);
            for (unsigned token = 0; token < 24; ++token) {
                const unsigned offset = (server * 24 + token) * 512;
                std::fill(local + offset, local + offset + 512, uint8_t(generation + token + server));
                transport.WriteNbi(outgoing, local + offset, local + offset, 512);
                transport.BatchCommit(outgoing);
            }
            // The upstream resets and reuses UB between peers, not payload GM.
            std::memset(scratch, 0xcc, sizeof(scratch));
            transport.Init({scratch}, sizeof(scratch));
        }
        assert(Model::pending.size() == 72 && Model::quiets == quietBefore);
        uint8_t fresh[128]{};
        transport.Retire({fresh});
        if (!Model::pending.empty() || Model::quiets != quietBefore + 3) {
            std::cerr << "generation=" << generation << " pending=" << Model::pending.size()
                      << " quiets=" << Model::quiets << " before=" << quietBefore << '\n';
            for (auto r : Model::pending) std::cerr << "peer=" << r.peer << " qp=" << r.qp << '\n';
        }
        assert(Model::pending.empty() && Model::quiets == quietBefore + 3);
        assert(std::memcmp(local, remote, 72 * 512) == 0);
    }
#endif
    std::cout << "SHMEM adapter: symmetric translation, per-core QP and batch completion passed\n";
}
