// Exercise the production ready protocol and Vector selector with checked
// CPU API stubs. Device cache/transport behavior still needs an NPU run.
#include "kernel_operator.h"
#include <array>
#include <cassert>
#include <future>
#include <memory>
#include <random>
#include <thread>
namespace AscendC {
template <class T>
void Muls(LocalTensor<T> d, LocalTensor<T> s, T x, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) d.p[i] = s.p[i] * x;
}
template <class T>
void Gather(LocalTensor<T> d, LocalTensor<T> s, LocalTensor<uint32_t> off, uint32_t base, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) std::memcpy(d.p + i, reinterpret_cast<uint8_t*>(s.p) + base + off.p[i], sizeof(T));
}
}  // namespace AscendC
#include "kernels/combine/low_latency/combine_ready.h"
#include "kernels/combine/low_latency/combine_ready_select.h"
#include "kernels/combine/low_latency/combine_notify_layout.h"
#include "kernels/dispatch/low_latency/device/ll_count_frame_transport.h"
using namespace AscendC;
using namespace CombineDedup;
constexpr uint32_t bankBytes = uint32_t(kSharedEpoch) + 8192U;
static_assert(COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET + kSharedReady >=
                  LlTransport::kCountTxBase + (1024U + 256U) * 32U,
              "Dispatch count TX must not overwrite shared flags");
static_assert(kSharedReady >= COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CORE_READY_OFFSET +
                                  DispatchDedup::kLlMaxRecvCoreNum * kCoreReadyRowBytes,
              "private notification table overlap");
template <uint32_t Group>
struct Reader {
    CombineGroupReady<Group> ready;
    std::array<float, 352> input{};
    std::array<float, 64> work{};
    std::array<float, 128> indices{};
    alignas(8) std::array<float, 8> result{};
    std::array<float, 288> publication{};
    void Init(DispatchDedup::CommArgs* a, uint32_t bank, uint32_t c, uint32_t recv)
    {
        ready.Init(a, uint64_t(bank) * bankBytes, 56U - recv + c, 56U - recv, recv, {input.data()}, {work.data()},
                   {indices.data()}, {result.data()}, {publication.data()}, {});
    }
};
template <uint32_t Group, uint32_t World = Group>
struct Cluster {
    std::array<std::vector<uint64_t>, World> memory;
    std::array<DispatchDedup::CommArgs, World> args{};
    std::array<std::vector<std::unique_ptr<Reader<Group>>>, World> readers;
    uint32_t recv;
    explicit Cluster(uint32_t count) : recv(count)
    {
        for (auto& mem : memory) mem.resize(bankBytes * 2U / 8U);
        for (uint32_t rank = 0; rank < World; ++rank) {
            auto& a = args[rank];
            a.rank = rank;
            a.rankSize = World;
            a.reserved0 = Group;
            for (uint32_t peer = rank / Group * Group; peer < (rank / Group + 1U) * Group; ++peer)
                a.peerMems[peer] = reinterpret_cast<uint8_t*>(memory[peer].data());
            for (uint32_t c = 0; c < recv; ++c) readers[rank].push_back(std::make_unique<Reader<Group>>());
        }
    }
    GlobalTensor<uint32_t> At(uint32_t rank, uint32_t bank, uint64_t off)
    {
        return {reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(memory[rank].data()) + bank * bankBytes + off)};
    }
    void Init(uint32_t bank)
    {
        for (uint32_t rank = 0; rank < World; ++rank)
            for (uint32_t c = 0; c < recv; ++c) readers[rank][c]->Init(&args[rank], bank, c, recv);
    }
    void Publish(uint32_t source)
    {
        for (uint32_t local = 0; local < Group; ++local)
            readers[source][recv - 8U + local % 8U]->ready.Publish(source / Group * Group + local);
    }
};
void NoLiveEvents()
{
    for (auto& direction : events)
        for (bool live : direction) assert(!live);
}
template <class F>
void WaitBlocked(F& future)
{
    assert(future.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
}
template <uint32_t Group>
void CheckShared(uint32_t recv)
{
    Cluster<Group> cluster(recv);
    // Alternate the real two banks, then reuse each repeatedly. A stale flag
    // must not satisfy the next invocation even after the first full mask.
    for (uint32_t round = 0; round < 8; ++round) {
        const uint32_t bank = round % 2, epoch = round / 2 + 1;
        cluster.Init(bank);
        for (auto& rank : cluster.readers)
            for (auto& reader : rank) assert(reader->ready.Poll() == 0U);
        for (uint32_t source = 0; source < Group - 1; ++source) cluster.Publish(source);
        const uint32_t partial = (1U << (Group - 1)) - 1U;
        for (auto& rank : cluster.readers)
            for (auto& reader : rank) assert(reader->ready.Poll() == partial);
        cluster.Publish(Group - 1);
        for (auto& rank : cluster.readers)
            for (auto& reader : rank) reader->ready.Retire();
        for (auto& rank : cluster.readers) rank[0]->ready.Release();
        for (uint32_t rank = 0; rank < Group; ++rank)
            for (uint32_t peer = 0; peer < Group; ++peer)
                assert(cluster.At(rank, bank, kSharedAck + peer * 32).GetValue(0) == epoch);
        NoLiveEvents();
    }
    // Fast producer cannot overwrite an unconsumed flag. The coordinator also
    // waits for the last consumer, including a core with no partial tasks.
    cluster.Init(0);
    for (uint32_t source = 0; source < Group; ++source) cluster.Publish(source);
    const uint32_t late = Group - 1;
    for (uint32_t rank = 0; rank < Group; ++rank)
        for (uint32_t c = 0; c < recv; ++c)
            if (rank != late || c != recv - 1) cluster.readers[rank][c]->ready.Retire();
    for (uint32_t rank = 0; rank < Group - 1; ++rank) cluster.readers[rank][0]->ready.Release();
    auto released = std::async(std::launch::async, [&] {
        cluster.readers[late][0]->ready.Release();
        NoLiveEvents();
    });
    WaitBlocked(released);
    auto& publisher = *cluster.readers[0][recv - 1];
    publisher.Init(&cluster.args[0], 0, recv - 1, recv);
    auto published = std::async(std::launch::async, [&] {
        publisher.ready.Publish(late);
        NoLiveEvents();
    });
    WaitBlocked(published);
    assert(cluster.At(late, 0, kSharedReady).GetValue(0) == 5U);
    cluster.readers[late][recv - 1]->ready.Retire();
    released.get();
    published.get();
    assert(cluster.At(late, 0, kSharedReady).GetValue(0) == 6U);
    NoLiveEvents();
}
template <uint32_t Group>
void CheckWrap()
{
    Cluster<Group> cluster(8);
    for (uint32_t bank = 0; bank < 2; ++bank)
        for (uint32_t rank = 0; rank < Group; ++rank) {
            for (uint32_t c = 0; c < 8; ++c)
                cluster.At(rank, bank, kSharedEpoch + (48 + c) * 32).SetValue(0, 0xfffffffeU);
            for (uint32_t peer = 0; peer < Group; ++peer) {
                cluster.At(rank, bank, kSharedAck + peer * 32).SetValue(0, 0xfffffffeU);
                cluster.At(rank, bank, kSharedReady + peer * 32).SetValue(0, 0xfffffffeU);
            }
        }
    for (uint32_t round = 0; round < 6; ++round) {
        cluster.Init(round % 2);
        for (auto& rank : cluster.readers)
            for (auto& reader : rank) assert(reader->ready.Poll() == 0U);
        for (uint32_t source = 0; source < Group; ++source) cluster.Publish(source);
        for (auto& rank : cluster.readers)
            for (auto& reader : rank) reader->ready.Retire();
        for (auto& rank : cluster.readers) rank[0]->ready.Release();
    }
}
template <uint32_t Group>
void CheckPrivate(uint32_t recv)
{
    Cluster<Group> cluster(recv);
    for (uint32_t bank = 0; bank < 2; ++bank) {
        cluster.Init(bank);
        const uint64_t offset = COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET;
        for (uint32_t source = 0; source < Group - 1; ++source)
            cluster.At(0, bank, offset + source * 32).SetValue(0, 0x3f800000U);
        auto& reader = *cluster.readers[0][0];
        assert(reader.ready.Poll() == (1U << (Group - 1)) - 1U);
        cluster.At(0, bank, offset + 512U).SetValue(0, 0x3f800000U);
        auto retiring = std::async(std::launch::async, [&] {
            reader.ready.Retire();
            NoLiveEvents();
        });
        WaitBlocked(retiring);
        assert(cluster.At(0, bank, offset).GetValue(0) == 0x3f800000U);
        cluster.At(0, bank, offset + (Group - 1) * 32).SetValue(0, 0x3f800000U);
        retiring.get();
        for (uint32_t word = 0; word < 128; ++word) assert(cluster.At(0, bank, offset).GetValue(word) == 0U);
        // Cleanup belongs to one receive core and one window, never a peer row.
        assert(cluster.At(0, bank, offset + 512U).GetValue(0) == 0x3f800000U);
    }
}
void CheckSelector()
{
    std::mt19937 rng(42);
    for (uint32_t group : {2U, 4U, 16U})
        for (uint32_t n : {0U, 1U, 63U, 64U, 65U, 1250U}) {
            const uint32_t rounded = (n + 63U) / 64U * 64U;
            std::vector<int32_t> deps(rounded + 8, kReadyTaskDone), work(rounded + 8, -1), out(rounded + 8, -1);
            std::vector<uint64_t> mask((rounded + 63U) / 64U + 8);
            for (uint32_t i = 0; i < n; ++i)
                deps[i] = int32_t(rng() % ((1U << group) - 1U) + 1U) | ((rng() % 2) ? kReadyTaskLocal : 0U);
            uint32_t observed = 0U, completed = 0U;
            for (uint32_t step = 0; step <= group; ++step) {
                if (step) observed |= 1U << (step - 1U);
                const uint32_t missing = ((1U << group) - 1U) ^ observed;
                for (bool remote : {true, false}) {
                    std::vector<int32_t> expected;
                    for (uint32_t i = 0; i < n; ++i)
                        if (!(uint32_t(deps[i]) & (missing | kReadyTaskDone | (remote ? kReadyTaskLocal : 0U))))
                            expected.push_back(i);
                    const auto count =
                        SelectReadyCombineTasks({deps.data()}, {work.data()}, {out.data()},
                                                {reinterpret_cast<uint8_t*>(mask.data())}, n, missing, remote);
                    assert(count == expected.size() && std::equal(expected.begin(), expected.end(), out.begin()));
                    for (uint32_t i = 0; i < count; ++i) {
                        deps[out[i]] = kReadyTaskDone;
                        ++completed;
                    }
                }
            }
            assert(completed == n);
            for (uint32_t i = rounded; i < work.size(); ++i) assert(work[i] == -1 && out[i] == -1);
            NoLiveEvents();
        }
}
void CheckFourHostReady()
{
    // Physical groups are 0..15,16..31,32..47,48..63. Remote peer pointers
    // stay null: readiness must never use UBMEM to notify a different server.
    auto storage = std::make_unique<Cluster<16, 64>>(28);
    auto& cluster = *storage;
    for (uint32_t bank = 0; bank < 2; ++bank) {
        cluster.Init(bank);
#if DEEPEP_LL_COMBINE_READY == 2
        for (uint32_t source = 0; source < 15; ++source) cluster.Publish(source);
        for (uint32_t rank = 0; rank < 64; ++rank)
            for (auto& reader : cluster.readers[rank]) assert(reader->ready.Poll() == (rank < 16 ? 0x7fffU : 0U));
        // The other three servers complete two generations while server 0
        // still waits for rank 15. Their ACKs cannot depend on a global group.
        for (uint32_t generation = 1; generation <= 2; ++generation) {
            if (generation == 2)
                for (uint32_t rank = 16; rank < 64; ++rank)
                    for (uint32_t c = 0; c < 28; ++c) cluster.readers[rank][c]->Init(&cluster.args[rank], bank, c, 28);
            for (uint32_t source = 16; source < 64; ++source) cluster.Publish(source);
            for (uint32_t rank = 16; rank < 64; ++rank) {
                for (auto& reader : cluster.readers[rank]) reader->ready.Retire();
                cluster.readers[rank][0]->ready.Release();
            }
        }
        cluster.Publish(15);
        for (uint32_t rank = 0; rank < 16; ++rank) {
            for (auto& reader : cluster.readers[rank]) reader->ready.Retire();
            cluster.readers[rank][0]->ready.Release();
        }
        for (uint32_t rank = 0; rank < 64; ++rank)
            for (uint32_t local = 0; local < 16; ++local)
                assert(cluster.At(rank, bank, kSharedAck + local * 32U).GetValue(0) == (rank < 16 ? 1U : 2U));
#else
        const uint64_t offset = COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET;
        for (uint32_t rank = 0; rank < 64; ++rank) {
            cluster.At(rank, bank, offset + (rank % 16) * 32U).SetValue(0, 0x3f800000U);
            assert(cluster.readers[rank][0]->ready.Poll() == (1U << (rank % 16)));
            for (uint32_t local = 0; local < 16; ++local)
                cluster.At(rank, bank, offset + local * 32U).SetValue(0, 0x3f800000U);
            cluster.readers[rank][0]->ready.Retire();
            for (uint32_t word = 0; word < 128; ++word) assert(cluster.At(rank, bank, offset).GetValue(word) == 0U);
        }
#endif
        NoLiveEvents();
    }
}
int main()
{
    CheckSelector();
    CheckFourHostReady();
#if DEEPEP_LL_COMBINE_READY == 2
    for (uint32_t recv : {8U, 28U, 36U}) {
        CheckShared<2>(recv);
        CheckShared<4>(recv);
        CheckShared<16>(recv);
    }
    CheckWrap<2>();
    CheckWrap<4>();
    CheckWrap<16>();
#else
    for (uint32_t recv : {8U, 28U, 36U}) {
        CheckPrivate<2>(recv);
        CheckPrivate<4>(recv);
        CheckPrivate<16>(recv);
    }
#endif
    return 0;
}
