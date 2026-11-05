// Copyright (c) 2026, Lu Lu
#pragma once

#include "combine_profile_device.h"
#include "ep_memory_server_dedup_combine_config.h"
#include "combine_server_dedup_select.h"
#include "combine_return_schedule.h"

namespace CombineDedup {
// M4 has not started computing yet. The caller lends either the idle FP32
// vectors or the not-yet-initialized payload arena. Owned rows are separate;
// no live payload, in-flight output or resident control storage is borrowed.
template<uint32_t Group>
class PartialTaskPlan {
public:
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* p)
    {
        profile_.Bind(p);
    }
    static constexpr uint32_t kMaxGroups = 16U;

    __aicore__ inline bool Prepare(AscendC::LocalTensor<uint8_t> scratch, uint32_t scratchBytes,
                                   AscendC::LocalTensor<int32_t> ownedRows,
                                   AscendC::GlobalTensor<int32_t> types,
                                   AscendC::GlobalTensor<int32_t> routes,
                                   uint32_t tokens, uint32_t world, uint32_t rank,
                                   uint32_t localCore, uint32_t recvCores)
    {
        using namespace AscendC;
        const uint32_t groups = world / Group;
        const uint32_t rows = EpCombineAlign(tokens, 64U);
        if (groups > kMaxGroups || EpCombineM4ScanBytes(rows) > scratchBytes) return false;
        profile_.Set(CombineProfile::RouteSnapshotBytes, EpCombineM4ScanBytes(rows));
        if (tokens == 0U) {
            for (uint32_t step = 0; step < groups; ++step) ends_[step] = 0U;
            return true;
        }
        const uint32_t vectorBytes = EpCombineAlign(rows * 4U, 256U);
        const uint32_t maskBytes = EpCombineAlign((rows + 7U) / 8U, 32U);
        auto table = scratch.ReinterpretCast<int32_t>();
        auto ranks = scratch[rows * 12U].ReinterpretCast<int32_t>();
        auto indices = scratch[rows * 12U + vectorBytes].ReinterpretCast<int32_t>();
        auto typeMask = scratch[rows * 12U + 2U * vectorBytes];
        auto mask = typeMask[maskBytes], upper = mask[maskBytes];
        const uint64_t loadStart = profile_.DetailClock();
        DataCopyPad(ranks, types, {1U, tokens * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        DataCopyPad(table, routes, {1U, tokens * 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        SetFlag<HardEvent::MTE2_V>(7U);
        WaitFlag<HardEvent::MTE2_V>(7U);
        profile_.Elapsed(CombineProfile::PlanLoadTicks, loadStart);
        profile_.Add(CombineProfile::RouteLoadBatches);
        profile_.Add(CombineProfile::RouteLoadedRows, tokens);
        profile_.Add(CombineProfile::RouteLoadBytes, uint64_t(tokens) * 16U);
        const uint64_t selectStart = profile_.DetailClock();
        Compares(typeMask, ranks, int32_t(2), CMPMODE::EQ, rows);
        CreateVecIndex(indices, int32_t(0), tokens);
        PipeBarrier<PIPE_V>();
        Muls(indices, indices, int32_t(12), tokens);
        PipeBarrier<PIPE_V>();
        Gather(ranks, table, indices.ReinterpretCast<uint32_t>(), 0U, tokens);
        PipeBarrier<PIPE_V>();
        CreateVecIndex(indices, int32_t(0), tokens);
        PipeBarrier<PIPE_V>();

        // Validate only FIRST_HOP source ranks: TERMINAL rows store slot IDs
        // in the same column. Validation and traversal match the baseline.
        uint64_t typed = 0U, valid = 0U;
        GatherMask(table, indices, typeMask.ReinterpretCast<uint32_t>(), true, tokens, {1U, 1U, 0U, 0U}, typed);
        ServerDedupRankMask(mask, upper, ranks, 0, int32_t(world), tokens);
        And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
            typeMask.ReinterpretCast<uint16_t>(), int32_t((tokens + 15U) / 16U));
        PipeBarrier<PIPE_V>();
        GatherMask(table, indices, mask.ReinterpretCast<uint32_t>(), true, tokens, {1U, 1U, 0U, 0U}, valid);
        SetFlag<HardEvent::V_S>(7U);
        WaitFlag<HardEvent::V_S>(7U);
        if (valid != typed) { Trap(); return false; }

        profile_.Set(CombineProfile::RouteFirstHopRows, valid);
        uint32_t filterRows = tokens;
#if DEEPEP_LL_COMBINE_READY == 8
        // The route triplets are dead after ranks have been gathered. Reuse
        // their three padded columns as selected IDs / compact IDs / ranks.
        // All Vector inputs/outputs are disjoint; ownedRows stays outside this
        // borrowed arena. Stable compaction preserves the original row order.
        filterRows = uint32_t(valid);
        if (!filterRows) {
            for (uint32_t step = 0; step < groups; ++step) ends_[step] = 0U;
            profile_.Elapsed(CombineProfile::PlanSelectTicks, selectStart);
            return true;
        }
        const uint64_t compactStart = profile_.DetailClock();
        auto compactIds = table[rows];
        auto compactRanks = table[2U * rows];
        Muls(compactIds, table, int32_t(1), filterRows);
        uint64_t compactCount = 0U;
        GatherMask(compactRanks, ranks, mask.ReinterpretCast<uint32_t>(), true, tokens,
                   {1U, 1U, 0U, 0U}, compactCount);
        PipeBarrier<PIPE_V>();
        indices = compactIds;
        ranks = compactRanks;
        profile_.Elapsed(CombineProfile::RouteCompactTicks, compactStart);
#endif
        uint32_t owner = 0U, count = 0U;
        for (uint32_t step = 0; step < groups; ++step) {
            profile_.Add(CombineProfile::RouteSelectPasses);
            profile_.Add(CombineProfile::RouteFilterRows, filterRows);
            const uint32_t server = CombineReturnServer<Group>(groups, rank / Group, step);
            ServerDedupRankMask(mask, upper, ranks, int32_t(server * Group), int32_t((server + 1U) * Group), filterRows);
#if DEEPEP_LL_COMBINE_READY != 8
            And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                typeMask.ReinterpretCast<uint16_t>(), int32_t((tokens + 15U) / 16U));
            PipeBarrier<PIPE_V>();
#endif
            uint64_t selected = 0U;
            GatherMask(table, indices, mask.ReinterpretCast<uint32_t>(), true, filterRows, {1U, 1U, 0U, 0U}, selected);
            SetFlag<HardEvent::V_S>(7U);
            WaitFlag<HardEvent::V_S>(7U);
            const uint32_t first = ServerDedupFirstOwned(localCore, owner, recvCores);
            owner = (owner + uint32_t(selected)) % recvCores;
            for (uint32_t task = first; task < uint32_t(selected); task += recvCores)
                ownedRows.SetValue(count++, table.GetValue(task));
            ends_[step] = count;
        }
        profile_.Elapsed(CombineProfile::PlanSelectTicks, selectStart);
        return true;
    }

    __aicore__ inline uint32_t End(uint32_t step) const { return ends_[step]; }

private:
    CombineProfile::View profile_;
    uint32_t ends_[kMaxGroups] = {};
};
}  // namespace CombineDedup
