// Execute the actual device helper with atomic memory and deferred RDMA mocks.
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>
#define __gm__
#define __ubuf__
#define __aicore__
constexpr int ACLSHMEM_TEAM_WORLD = 0, ACLSHMEM_TRANSPORT_MTE = 1, ACLSHMEM_TRANSPORT_ROCE = 4;
struct alignas(64) Bit { int32_t value = 0; };
constexpr int ACLSHMEM_SYNCBIT_SIZE = sizeof(Bit), SYNC_ARRAY_SIZE = 32 * sizeof(Bit);
struct Team { int mype, size; int pe_mapping[64]; };
struct State { Team* team_pools[1]; uint8_t topo_list[64]; };
struct Rank {
    Team team{};
    State state{};
    Bit pool[32], counter;
    std::atomic<int> arrived{0};
};
std::array<Rank, 64> ranks;
thread_local int me;
thread_local int32_t* pendingDst = nullptr;
thread_local int32_t* pendingSrc = nullptr;
std::atomic<int> rdmaCalls{0}, mteCalls{0}, aheadReads{0};
int hostSize;
bool nullMapping = false;
namespace AscendC {
enum class CacheMode { CACHE_MODE_DISABLE };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template<class T> struct GlobalTensor {
    T* p;
    void SetGlobalBuffer(T* addr) { p = addr; }
    void SetL2CacheHint(CacheMode) {}
    void SetValue(int, T value) { std::atomic_ref<T>(*p).store(value); }
    T GetValue(int) {
        T value = std::atomic_ref<T>(*p).load();
        if (p != &ranks[me].counter.value && value > ranks[me].counter.value) ++aheadReads;
        return value;
    }
};
template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
inline uint64_t GetSystemCycle() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
[[noreturn]] inline void Trap() { throw std::runtime_error("device trap"); }
using std::printf;
}
State* aclshmemi_get_state() { return &ranks[me].state; }
Bit* aclshmemi_get_team_sync_pool(int) { return ranks[me].pool; }
Bit* aclshmemi_get_team_sync_counter(int) { return &ranks[me].counter; }
int32_t* mappedSlot(int32_t* local, int peer) {
    for (int i = 0; i < 32; ++i)
        if (local == &ranks[me].pool[i].value) return &ranks[peer].pool[i].value;
    assert(false); return nullptr;
}
void* aclshmem_ptr(void* p, int peer) {
    assert(me / hostSize == peer / hostSize);
    ++mteCalls;
    return nullMapping ? nullptr : mappedSlot(static_cast<int32_t*>(p), peer);
}
void aclshmemx_roce_qp_put_nbi(uint8_t* dst, uint8_t* src, uint8_t*, uint32_t size,
                              int peer, unsigned qp, unsigned event) {
    assert(size == 4 && qp == 0 && event == 0 && !pendingSrc);
    // Intra-host MTE must win even when SHMEM also reports the RoCE bit.
    assert(me / hostSize != peer / hostSize);
    assert(src == reinterpret_cast<uint8_t*>(&ranks[me].counter));
    pendingSrc = reinterpret_cast<int32_t*>(src);
    pendingDst = mappedSlot(reinterpret_cast<int32_t*>(dst), peer);
    ++rdmaCalls;
}
void aclshmemx_roce_qp_quiet(int, unsigned qp, uint8_t*, unsigned event) {
    assert(pendingSrc && qp == 0 && event == 0);
    std::this_thread::yield();
    std::atomic_ref<int32_t>(*pendingDst).store(*pendingSrc);
    pendingSrc = pendingDst = nullptr;
}
#include "entry_sync_device.h"

void setup(int world, int perHost) {
    hostSize = perHost;
    rdmaCalls = mteCalls = aheadReads = 0;
    nullMapping = false;
    for (int i = 0; i < world; ++i) {
        auto& r = ranks[i];
        r.team.mype = i; r.team.size = world; r.state.team_pools[0] = &r.team;
        r.counter.value = 17; r.arrived = 0;
        for (auto& slot : r.pool) slot.value = 17;
        for (int j = 0; j < world; ++j) {
            r.team.pe_mapping[j] = j;
            r.state.topo_list[j] = ACLSHMEM_TRANSPORT_ROCE |
                (i / perHost == j / perHost ? ACLSHMEM_TRANSPORT_MTE : 0);
        }
    }
}
void repeated(int world, int perHost) {
    setup(world, perHost);
    constexpr int rounds = 80;
    std::vector<std::thread> workers;
    for (int i = 0; i < world; ++i) workers.emplace_back([=] {
        me = i;
        uint8_t scratch[512];
        for (int iteration = 1; iteration <= rounds; ++iteration) {
            if (i == iteration % world)
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            ranks[i].arrived = iteration;
            ll_bench_sync_world(scratch, 30000000000ULL);
            for (int peer = 0; peer < world; ++peer) assert(ranks[peer].arrived >= iteration);
            assert(!pendingSrc && ranks[i].counter.value == 17 + iteration);
        }
    });
    for (auto& worker : workers) worker.join();
    assert((rdmaCalls > 0) == (world > perHost));
    if (world > 1 && perHost > 1) assert(mteCalls > 0);
}
int main() {
    for (int world : {1, 2, 4, 8, 16}) repeated(world, world);
    for (auto topology : {std::pair{8, 4}, {32, 16}, {64, 16}, {6, 2}})
        repeated(topology.first, topology.second);
    // Missing participants, unsupported paths and corrupt generation fail closed.
    for (int failure = 0; failure < 4; ++failure) {
        setup(2, 2); me = 0;
        if (failure == 1) ranks[0].state.topo_list[1] = 0;
        if (failure == 2) ranks[0].counter.value = INT32_MAX - 1;
        if (failure == 3) nullMapping = true;
        uint8_t scratch[512];
        bool trapped = false;
        try { ll_bench_sync_world(scratch, 1000000); }
        catch (const std::runtime_error&) { trapped = true; }
        assert(trapped);
    }
}
