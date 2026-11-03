// Benchmark-only arrival synchronization. No change to the installed SHMEM SDK.
// The fix_1825_sync_error branch keeps NPU-domain synchronization off RoCE:
// https://gitcode.com/vallenChen/shmem/commit/8aacd2bb072bca900d75ea79ed185da43ac3be7c
// Extend that transport choice per dissemination edge for a cross-host WORLD.
#pragma once
#include <climits>

__aicore__ inline void ll_bench_sync_store(__gm__ int32_t* address, int32_t value)
{
    AscendC::GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address);
    signal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    signal.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                    AscendC::DcciDst::CACHELINE_OUT>(signal);
}

__aicore__ inline int32_t ll_bench_sync_load(__gm__ int32_t* address)
{
    AscendC::GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address);
    signal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                    AscendC::DcciDst::CACHELINE_OUT>(signal);
    return signal.GetValue(0);
}

__aicore__ inline void ll_bench_sync_world(__ubuf__ uint8_t* scratch, uint64_t waitTicks)
{
    // Exactly one AIV owns QP0 and the WORLD generation. All benchmark work on
    // this device has completed before this kernel; no generic quiet/barrier
    // may touch a nonexistent RoCE QP between the two dies of one 1825.
    auto state = aclshmemi_get_state();
    auto team = state->team_pools[ACLSHMEM_TEAM_WORLD];
    auto pool = aclshmemi_get_team_sync_pool(ACLSHMEM_TEAM_WORLD);
    auto counter = reinterpret_cast<__gm__ int32_t*>(aclshmemi_get_team_sync_counter(ACLSHMEM_TEAM_WORLD));
    int index = -1;
    for (int i = 0; i < team->size; ++i)
        if (team->pe_mapping[i] == team->mype) index = i;
    if (index < 0 || !waitTicks) { AscendC::Trap(); return; }
    const int32_t previous = ll_bench_sync_load(counter);
    if (previous < 0 || previous >= INT32_MAX - 1) { AscendC::Trap(); return; }
    const int32_t generation = previous + 1;
    ll_bench_sync_store(counter, generation);

    int slot = 0;
    for (int shift = 1; shift < team->size; shift <<= 1, ++slot) {
        if (slot >= SYNC_ARRAY_SIZE / ACLSHMEM_SYNCBIT_SIZE) { AscendC::Trap(); return; }
        const int peer = team->pe_mapping[(index + shift) % team->size];
        const int predecessor = team->pe_mapping[(index - shift + team->size) % team->size];
        auto localSlot = reinterpret_cast<__gm__ int32_t*>(pool + slot);
        const uint8_t transport = state->topo_list[peer];
        if (transport & ACLSHMEM_TRANSPORT_MTE) {
            auto mapped = reinterpret_cast<__gm__ int32_t*>(aclshmem_ptr(localSlot, peer));
            if (!mapped) { AscendC::Trap(); return; }
            ll_bench_sync_store(mapped, generation);
        } else if (transport & ACLSHMEM_TRANSPORT_ROCE) {
            // Both addresses are LOCAL symmetric addresses, translated by SHMEM.
            // Retire the source before the next invocation changes the counter.
            aclshmemx_roce_qp_put_nbi(reinterpret_cast<__gm__ uint8_t*>(localSlot),
                reinterpret_cast<__gm__ uint8_t*>(counter), scratch, sizeof(int32_t), peer, 0U, 0U);
            aclshmemx_roce_qp_quiet(peer, 0U, scratch, 0U);
        } else {
            AscendC::printf("[LL entry sync] rank=%d peer=%d unsupported transport=%u\n",
                            team->mype, peer, unsigned(transport));
            AscendC::Trap();
            return;
        }
        const uint64_t begin = AscendC::GetSystemCycle();
        while (true) {
            const int32_t seen = ll_bench_sync_load(localSlot);
            // A faster predecessor can already enter the next barrier. It
            // cannot advance two generations without this rank participating.
            if (seen == generation || seen == generation + 1) break;
            if (AscendC::GetSystemCycle() - begin >= waitTicks) {
                AscendC::printf("[LL entry sync TIMEOUT] rank=%d predecessor=%d slot=%d expected=%d seen=%d\n",
                                team->mype, predecessor, slot, generation, seen);
                AscendC::Trap();
                return;
            }
        }
    }
}
