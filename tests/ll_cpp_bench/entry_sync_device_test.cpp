// Host model executes the actual barrier helper, mocking only SDK primitives.
// It cannot validate device cache/coherence or the RoCE hardware implementation.
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>
#define __gm__
#define __ubuf__
#define ACLSHMEM_DEVICE inline
constexpr int ACLSHMEM_TEAM_WORLD = 0, ACLSHMEM_TRANSPORT_MTE = 1, ACLSHMEM_TRANSPORT_ROCE = 4;
struct alignas(64) Bit { int32_t value = 0; };
struct Team { int team_idx = 0, mype = 0, start = 0, stride = 1, size = 0; };
struct State { Team* team_pools[1]; uint8_t topo_list[64]; };
struct Rank {
    Team team;
    State state{};
    Bit pool[32], counter;
    std::atomic<int> arrived{0};
};
std::array<Rank, 64> ranks;
thread_local int me;
std::atomic<int> rdmaCalls{0}, mteCalls{0}, aheadReads{0};
State* aclshmemi_get_state() { return &ranks[me].state; }
Bit* aclshmemi_get_team_sync_pool(int team) { assert(team == 0); return ranks[me].pool; }
Bit* aclshmemi_get_team_sync_counter(int team) { assert(team == 0); return &ranks[me].counter; }
int32_t aclshmemi_load(int32_t* p) { return std::atomic_ref<int32_t>(*p).load(); }
void aclshmemi_store(int32_t* p, int32_t v) { std::atomic_ref<int32_t>(*p).store(v); }
void dcci_cacheline(uint8_t*) {}
int32_t* mapped(int32_t* p, int peer) {
    for (int slot = 0; slot < 32; ++slot)
        if (p == &ranks[me].pool[slot].value) {
            assert((1U << slot) < unsigned(ranks[me].team.size));
            return &ranks[peer].pool[slot].value;
        }
    assert(false); return nullptr;
}
void* aclshmem_ptr(void* p, int peer) {
    assert(ranks[me].state.topo_list[peer] & ACLSHMEM_TRANSPORT_MTE);
    ++mteCalls;
    return mapped(static_cast<int32_t*>(p), peer);
}
void aclshmemi_signal_set(int32_t* p, int32_t generation) { aclshmemi_store(p, generation); }
void aclshmemi_highlevel_signal_set(int32_t* dst, int32_t* src, int peer, uint8_t*, uint32_t event) {
    assert(ranks[me].state.topo_list[peer] == ACLSHMEM_TRANSPORT_ROCE && event == 0);
    assert(src == &ranks[me].counter.value);
    ++rdmaCalls;
    // Simulate SDK write+quiet: source is consumed before this call returns.
    std::this_thread::yield();
    aclshmemi_store(mapped(dst, peer), aclshmemi_load(src));
}
void aclshmemi_signal_wait_until_eq_for_barrier(int32_t* p, int32_t generation) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        auto value = aclshmemi_load(p);
        if (value == generation || value == generation + 1) {
            if (value == generation + 1) ++aheadReads;
            return;
        }
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
}
[[noreturn]] void aclshmemi_kernel_abort(const char*, int) { throw std::runtime_error("unreachable peer"); }
#include "entry_sync_device.h"

void setup(int world, int perHost, bool pairOnly) {
    rdmaCalls = mteCalls = aheadReads = 0;
    for (int i = 0; i < world; ++i) {
        auto& r = ranks[i];
        r.team = {0, i, 0, 1, world}; r.state.team_pools[0] = &r.team;
        r.counter.value = 17; r.arrived = 0;
        for (auto& slot : r.pool) slot.value = 17;
        for (int j = 0; j < world; ++j) {
            // Same 1825 pair: MTE only; other same-host peers may advertise both.
            r.state.topo_list[j] = i / 2 == j / 2 ? ACLSHMEM_TRANSPORT_MTE :
                ((i / perHost == j / perHost && !pairOnly) ?
                 ACLSHMEM_TRANSPORT_MTE | ACLSHMEM_TRANSPORT_ROCE : ACLSHMEM_TRANSPORT_ROCE);
        }
    }
}
void repeated(int world, int perHost, bool pairOnly) {
    setup(world, perHost, pairOnly);
    constexpr int iterations = 80;
    std::vector<std::thread> workers;
    for (int rank = 0; rank < world; ++rank) workers.emplace_back([=] {
        me = rank;
        uint8_t scratch[512];
        for (int iteration = 1; iteration <= iterations; ++iteration) {
            if (rank == iteration % world) std::this_thread::sleep_for(std::chrono::microseconds(50));
            ranks[rank].arrived = iteration;
            ll_bench_sync_world(scratch);
            // No peer may leave the barrier before all ranks entered it.
            for (int peer = 0; peer < world; ++peer) assert(ranks[peer].arrived >= iteration);
            assert(ranks[rank].counter.value == 17 + iteration);
        }
    });
    for (auto& worker : workers) worker.join();
    assert((rdmaCalls > 0) == (world > perHost || (pairOnly && world > 2)));
    assert(mteCalls > 0);
}
int main() {
    for (int world : {2, 4, 8, 16}) repeated(world, world, false);
    for (auto topology : {std::pair{8, 4}, {32, 16}, {64, 16}}) {
        repeated(topology.first, topology.second, false);
        repeated(topology.first, topology.second, true);
    }
    // A signal one generation ahead is legal under SDK barrier semantics.
    setup(2, 2, false); me = 0;
    ranks[0].pool[0].value = 19;
    ll_bench_sync_world(nullptr);
    assert(aheadReads == 1 && ranks[1].pool[0].value == 18);
    setup(2, 2, false); ranks[0].state.topo_list[1] = 0;
    bool rejected = false;
    try { ll_bench_sync_world(nullptr); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
