// Copyright (c) 2026, Lu Lu
// Modified by ryan_li 2026

#ifndef COMBINE_DEDUP_EP_SERVER_DEDUP_SOURCE_AGGREGATE_H
#define COMBINE_DEDUP_EP_SERVER_DEDUP_SOURCE_AGGREGATE_H

#include "kernel_operator.h"
#include "combine_profile_device.h"
#include "combine_poll.h"
#include "combine_cache_policy.h"
#include "combine_ready_layout.h"
#ifdef DEEPEP_LL_COMBINE_SIMT_FLAG
#include "combine_simt.h"
#endif
#include "ep_memory_server_dedup_combine_config.h"
#include "../../dispatch/low_latency/common/dispatch_dedup_types.h"

namespace CombineDedup {
using namespace AscendC;

// 中文审计：M5按源token及权重聚合回传结果并写yOut；非去重路径由InitDirectShared进入M6。
// 修改输入等待、输出槽复用或clear flags时，需同时审计上层DrainSharedOutputs/DrainM6收尾。
// M5 inherits M4 output slots and their completion events. Metadata/compute/input
// are handed off after MTE2/Vector completion, while M4 MTE3 may still run.
class ServerDedupSourceAggregate {
public:
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
    __aicore__ inline void SetSourceRoutes(GM_ADDR indices, uint32_t rank, uint32_t expertsPerRank,
                                          uint32_t ranksPerHost)
    {
        sourceIndices_ = indices;
        rank_ = rank;
        expertsPerRank_ = expertsPerRank;
        ranksPerHost_ = ranksPerHost;
    }
#endif
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* profile) { profile_.Bind(profile); }
    __aicore__ inline void Init(LocalTensor<uint8_t> scratch, GM_ADDR result, GM_ADDR masks, GM_ADDR weights,
                                GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t k, uint8_t dtype, uint32_t participant,
                                uint32_t participants, const EpCombineM45WorkspaceLayout& layout, uint32_t partialTasks,
                                bool enableDedup, bool uncached=false)
    {
        outputBytes_ = layout.outputBytes;
        finalLayout_ = EpCombineM5Layout(layout, partialTasks);
        inputLanes_ = finalLayout_.inputLanes;
        sharedOutputLanes_ = layout.outputLanes;
        outputLanes_ = layout.finalOutputLanes;
        if (inputLanes_ == 0U || outputLanes_ == 0U) {
            Trap();
            return;
        }
        InitCommon(scratch, result, masks, weights, yOut, bs, h, k, dtype, participant, participants,
                   layout.phaseOffset, layout.computeBytes, layout.outputOffset, enableDedup, uncached);
    }

    // Auto direct aggregation shares the M5 resource layout without inheriting
    // any M4 sends or events. Only the slots actually used for outputs are seeded.
    __aicore__ inline void InitDirectShared(LocalTensor<uint8_t> scratch, GM_ADDR result, GM_ADDR weights, GM_ADDR yOut,
                                            uint32_t bs, uint32_t h, uint32_t k, uint8_t dtype, uint32_t participant,
                                            uint32_t participants, const EpCombineM45WorkspaceLayout& layout, bool uncached=false)
    {
        finalLayout_ = EpCombineM5Layout(layout, 0U);
        inputLanes_ = finalLayout_.inputLanes;
        outputLanes_ = layout.finalOutputLanes;
        outputBytes_ = layout.outputBytes;
        if (inputLanes_ == 0U || outputLanes_ == 0U) {
            Trap();
            return;
        }
        for (uint32_t lane = 0; lane < outputLanes_; ++lane) SetFlag<HardEvent::MTE3_V>(finalLayout_.outputSlots[lane]);
        InitCommon(scratch, result, nullptr, weights, yOut, bs, h, k, dtype, participant, participants,
                   layout.phaseOffset, layout.computeBytes, layout.outputOffset, false, uncached);
    }

    __aicore__ inline void DrainM6()
    {
        for (uint32_t lane = 0; lane < outputLanes_; ++lane)
            WaitFlag<HardEvent::MTE3_V>(finalLayout_.outputSlots[lane]);
    }

    __aicore__ inline uint32_t Process()
    {
        profile_.Set(CombineProfile::FinalTokens, count_);
        profile_.Set(CombineProfile::InputLanes, inputLanes_);
        profile_.Set(CombineProfile::OutputLanes, outputLanes_);
        if (count_ == 0U) return 0U;
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) SetFlag<HardEvent::V_MTE2>(lane);
#ifdef DEEPEP_LL_COMBINE_BATCH_FINAL
        return ProcessBatched();
#endif
#if defined(DEEPEP_LL_COMBINE_EARLY_REDUCE) || defined(DEEPEP_LL_COMBINE_LOCAL_FIRST)
        if (incrementalReduce_) return ProcessEarly();
#endif
        uint32_t remaining = count_, scan = 0U, completed = 0U;
        while (remaining != 0U) {
            const uint32_t task = pending_.GetValue(scan);
            const uint32_t token = task & 0xffffU;
#ifdef DEEPEP_LL_COMBINE_PROGRESS_PROBE
            const bool ready = ProbeProgress(token, task >> 16U, scan);
#else
            const bool ready = tinyProbe_ ? ProbeTiny(token) : Probe(token, task >> 16U);
#endif
            if (!ready) {
                PauseAfterFlagMiss(profile_);
                if (++scan == remaining) scan = 0U;
                continue;
            }
            // No further probes after this point: clear every original token's
            // complete K*N flags, including mask-inactive slots. This source is
            // independent of the final token's in-flight payload and yOut.
            if (remaining == 1U) ClearFlags();
            const uint32_t mask = enableDedup_ ? masks_.GetValue(token - begin_) : 0U;
            if (dtype_ == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16)
                Aggregate<half>(token, mask, completed);
            else
                Aggregate<bfloat16_t>(token, mask, completed);
            pending_.SetValue(scan, pending_.GetValue(remaining - 1U));
            --remaining;
            ++completed;
            if (scan == remaining) scan = 0U;  // re-probe the swapped real TokenId
        }
        const uint64_t drainStart = profile_.DetailClock();
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) WaitFlag<HardEvent::V_MTE2>(lane);
        profile_.Elapsed(CombineProfile::FinalInputDrainTicks, drainStart);
        return retiredOutputs_;
    }

private:
    __aicore__ inline void InitCommon(LocalTensor<uint8_t> scratch, GM_ADDR result, GM_ADDR masks, GM_ADDR weights,
                                      GM_ADDR yOut, uint32_t bs, uint32_t h, uint32_t k, uint8_t dtype,
                                      uint32_t participant, uint32_t participants, uint32_t phaseOffset,
                                      uint32_t computeBytes, uint32_t outputOffset, bool enableDedup, bool uncached)
    {
        enableDedup_ = enableDedup;
        uncached_ = uncached;
        scratch_ = scratch[phaseOffset];
        h_ = h;
        k_ = k;
        dtype_ = dtype;
#if defined(DEEPEP_LL_COMBINE_EARLY_REDUCE) || defined(DEEPEP_LL_COMBINE_LOCAL_FIRST)
        incrementalReduce_ = EpCombineIncrementalReduce(bs);
        computeElements_ = computeBytes / sizeof(float);
#endif
        const uint32_t base = bs / participants, extra = bs % participants;
        count_ = base + (participant < extra ? 1U : 0U);
        begin_ = participant * base + (participant < extra ? participant : extra);
        // One token per core needs only a register-sized pending mask. No UB
        // growth, accumulator reassociation, or cross-token scratch aliasing.
        tinyProbe_ = kCombineTinyProbe && bs <= 24U && count_ == 1U && k <= 8U;
        profile_.Set(CombineProfile::TinyPolicy, (kCombineTinySetup && bs <= 24U ? 1U : 0U) | (tinyProbe_ ? 2U : 0U));
        blocks_ = (h * 2U + 479U) / 480U;
        packetBytes_ = blocks_ * 512U;
        // The shared Host budget uses the largest contiguous Recv-core segment.
        const uint32_t owned = EpCombineM5TokenCount(bs, participants);
#ifdef DEEPEP_LL_COMBINE_BATCH_FINAL
        batchIncremental_ = owned <= 2U;
#endif
        masks_ = Allocate<uint32_t>(EpCombineAlign(owned * 4U, 32U));
        pending_ = Allocate<uint32_t>(EpCombineAlign(owned * 4U, 32U));
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
        localMasks_ = Allocate<uint32_t>(EpCombineAlign(owned * 4U, 32U));
#endif
        weights_ = Allocate<float>(EpCombineAlign(owned * k * 4U, 32U));
        flags_ = Allocate<float>(EpCombineM5SnapshotBytes(h, k, owned));
        work_ = Allocate<float>(EpCombineM5WorkBytes(h, k));
        probeSum_ = Allocate<float>(32U);
#ifdef DEEPEP_LL_COMBINE_BATCH_FINAL
        routeMask_ = Allocate<uint8_t>(EpCombineM5CompareBytes(k, owned));
        routeUpper_ = Allocate<uint8_t>(EpCombineM5CompareBytes(k, owned));
#endif
        cast_ = scratch.ReinterpretCast<float>();
        product_ = scratch[computeBytes].ReinterpretCast<float>();
        sum_ = scratch[2U * computeBytes].ReinterpretCast<float>();
        inputs_ = scratch.ReinterpretCast<uint16_t>();
        outputs_ = scratch[outputOffset].ReinterpretCast<uint16_t>();
        result_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(result));
        resultFlags_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result));
        if (uncached && !kCombineResultL2) {
            result_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            resultFlags_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(yOut));
        if (count_ == 0U) return;
        GlobalTensor<uint32_t> maskGm;
        GlobalTensor<float> weightGm;
        if (enableDedup_) maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(masks));
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
        const uint64_t metaStart = profile_.DetailClock();
        if (enableDedup_) DataCopyPad(masks_, maskGm[begin_], {1U, count_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        DataCopyPad(weights_, weightGm[uint64_t(begin_) * k], {1U, count_ * k * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0.0f});
        Sync<HardEvent::MTE2_S>();
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
        if (sourceIndices_ == nullptr || expertsPerRank_ == 0U || ranksPerHost_ == 0U) { Trap(); return; }
        GlobalTensor<int32_t> routes;
        routes.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIndices_));
#ifdef DEEPEP_LL_COMBINE_BATCH_FINAL
        // One route DMA and vector range test for all tokens owned by this core.
        // flags_ is idle until polling; its shared Host budget covers this view.
        const uint64_t routeStart = profile_.DetailClock();
        auto ids = flags_.ReinterpretCast<int32_t>();
        DataCopyPad(ids, routes[uint64_t(begin_) * k_], {1U, count_ * k_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        Sync<HardEvent::MTE2_V>();
        const int32_t lower = int32_t(rank_ / ranksPerHost_ * ranksPerHost_ * expertsPerRank_);
        Compares(routeMask_, ids, lower, CMPMODE::GE, count_ * k_);
        Compares(routeUpper_, ids, lower + int32_t(ranksPerHost_ * expertsPerRank_), CMPMODE::LT, count_ * k_);
        PipeBarrier<PIPE_V>();
        And(routeMask_.ReinterpretCast<uint16_t>(), routeMask_.ReinterpretCast<uint16_t>(),
            routeUpper_.ReinterpretCast<uint16_t>(), EpCombineM5CompareBytes(k_, count_) / 2U);
        Sync<HardEvent::V_S>();
        auto bits = routeMask_.ReinterpretCast<uint32_t>();
        for (uint32_t p = 0U; p < count_; ++p) {
            const uint32_t bit = p * k_, shift = bit % 32U;
            uint32_t local = bits.GetValue(bit / 32U) >> shift;
            if (shift && shift + k_ > 32U) local |= bits.GetValue(bit / 32U + 1U) << (32U - shift);
            localMasks_.SetValue(p, local & (k_ == 32U ? 0xffffffffU : (1U << k_) - 1U));
        }
        profile_.Elapsed(CombineProfile::BatchRouteTicks, routeStart);
#else
        // Reuse probe scratch before its first Vector use. Each active return
        // keeps its original top-k slot, including a remote FIRST_HOP partial.
        auto ids = work_.ReinterpretCast<int32_t>();
        for (uint32_t p = 0U; p < count_; ++p) {
            DataCopyPad(ids, routes[uint64_t(begin_ + p) * k_], {1U, k_ * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            Sync<HardEvent::MTE2_S>();
            uint32_t local = 0U;
            for (uint32_t slot = 0U; slot < k_; ++slot) {
                const int32_t expert = ids.GetValue(slot);
                if (expert >= 0 && uint32_t(expert) / expertsPerRank_ / ranksPerHost_ == rank_ / ranksPerHost_)
                    local |= 1U << slot;
            }
            localMasks_.SetValue(p, local);
            // Finish Scalar reads before reusing this route snapshot for DMA.
            Sync<HardEvent::S_MTE2>();
        }
#endif
#endif
        profile_.Elapsed(CombineProfile::FinalMetaLoadTicks, metaStart);
        const uint64_t pendingStart = profile_.DetailClock();
        for (uint32_t p = 0; p < count_; ++p) {
            const uint32_t mask = enableDedup_ ? masks_.GetValue(p) : 0U;
            if (enableDedup_ && (mask == 0U || (k_ < 32U && (mask >> k_) != 0U))) {
                Trap();
                return;
            }
#ifdef DEEPEP_LL_COMBINE_PROGRESS_PROBE
            // High half stores the first contribution not yet confirmed.
            pending_.SetValue(p, begin_ + p);
#else
            const uint32_t expected =
                (enableDedup_ ? uint32_t(GetBitCount<1>(uint64_t(mask))) : k_) * blocks_;
            // BuildLayout admits B <= 65536; the existing flag DMA bound is
            // <=4095. Cache both fields in the original uint32 task slot.
            pending_.SetValue(p, (begin_ + p) | (expected << 16U));
#endif
        }
        if (tinyProbe_) {
            tinyPendingMask_ = enableDedup_ ? masks_.GetValue(0U) : (1U << k_) - 1U;
            tinyActiveSlots_ = uint32_t(GetBitCount<1>(uint64_t(tinyPendingMask_)));
        }
        profile_.Elapsed(CombineProfile::FinalPendingBuildTicks, pendingStart);
    }
    template <class T>
    __aicore__ inline LocalTensor<T> Allocate(uint32_t bytes)
    {
        auto tensor = scratch_[used_].template ReinterpretCast<T>();
        used_ += bytes;
        return tensor;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        SetFlag<E>(7U);
        WaitFlag<E>(7U);
    }
    __aicore__ inline bool ProbeTiny(uint32_t token)
    {
        const uint64_t start = profile_.DetailClock();
        const uint32_t pending = tinyPendingMask_;
        const uint32_t slots = uint32_t(GetBitCount<1>(uint64_t(pending)));
        // Coalesce adjacent pending slots into a single strided DMA. Retain
        // their original snapshot positions for tail diagnostics. Every 512B
        // flag (including the last block) remains part of the readiness test.
        for (uint32_t slot = 0U; slot < k_;) {
            if (((pending >> slot) & 1U) == 0U) { ++slot; continue; }
            const uint32_t first = slot++;
            while (slot < k_ && ((pending >> slot) & 1U)) ++slot;
            auto src = resultFlags_[(uint64_t(token) * k_ + first) * packetBytes_ / 4U + 120U];
            DataCopy(flags_[first * blocks_ * 8U], src, {uint16_t((slot - first) * blocks_), 1U, 15U, 0U});
        }
        Sync<HardEvent::MTE2_V>();
        profile_.Elapsed(CombineProfile::ProbeLoadTicks, start);
        const uint64_t check = profile_.DetailClock();
        // Vector checks all blocks of each contribution. At most eight aligned
        // sums fit in the existing >=256B work_ buffer. M4 has retired its
        // compute reader; M5 has not loaded payloads, so cast_ is safe scratch.
        for (uint32_t slot = 0U; slot < k_; ++slot) if ((pending >> slot) & 1U) {
            ReduceSum(work_[slot * 8U], flags_[slot * blocks_ * 8U], cast_, 1U, blocks_, 1U);
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_S>();
        for (uint32_t slot = 0U; slot < k_; ++slot)
            if (((pending >> slot) & 1U) && work_.GetValue(slot * 8U) == float(blocks_))
                tinyPendingMask_ &= ~(1U << slot);
        const bool ready = tinyPendingMask_ == 0U;
        profile_.Elapsed(CombineProfile::ProbeCheckTicks, check);
        profile_.Add(CombineProfile::FinalSparseProbeCalls);
        profile_.Add(CombineProfile::FinalSparseSkippedFlags, (tinyActiveSlots_ - slots) * blocks_);
        profile_.Add(CombineProfile::FinalSparseConfirmedSlots, slots - uint32_t(GetBitCount<1>(uint64_t(tinyPendingMask_))));
        profile_.Probe(ready, token, 0xffffffffU, slots * blocks_, start);
        InspectMissing(token, ready, 0U, k_);
        // Retire Scalar reads before reusing the per-slot sums / snapshot.
        Sync<HardEvent::S_V>();
        Sync<HardEvent::S_MTE2>();
        return ready;
    }
    __aicore__ inline bool Probe(uint32_t token, uint32_t expected)
    {
        // The previous probe's V_S retired its flag-buffer reader. This UB is
        // independent of payload compute/output, so do not wait for that Vector
        // work before issuing the next flag read. Inactive slot flags stay zero.
        const uint64_t start = profile_.DetailClock();
        const uint32_t flags = k_ * blocks_;
        // Include each packet's final flag block: RoCE SO makes its arrival
        // a whole-message prerequisite. MTE and mask-inactive slot rules stay unchanged.
        auto src = resultFlags_[uint64_t(token) * k_ * packetBytes_ / 4U + 120U];
        DataCopy(flags_, src, {uint16_t(flags), 1U, 15U, 0U});
        Sync<HardEvent::MTE2_V>();
        profile_.Elapsed(CombineProfile::ProbeLoadTicks, start);
        const uint64_t checkStart = profile_.DetailClock();
#ifdef DEEPEP_LL_COMBINE_SIMT_FLAG
        const uint32_t active = enableDedup_ ? masks_.GetValue(token - begin_)
            : (k_ == 32U ? 0xffffffffU : (1U << k_) - 1U);
        auto* work = reinterpret_cast<__ubuf__ uint32_t*>(work_.GetPhyAddr());
        auto* summary = reinterpret_cast<__ubuf__ uint32_t*>(probeSum_.GetPhyAddr());
        asc_vf_call<ResultFlagsVf>(dim3(32), reinterpret_cast<__ubuf__ float*>(flags_.GetPhyAddr()),
                                  work, k_, blocks_, active);
        PipeBarrier<PIPE_V>();
        asc_vf_call<ResultFlagsSummaryVf>(dim3(32), work, summary, k_, blocks_);
        Sync<HardEvent::V_S>();
        auto result = probeSum_.ReinterpretCast<uint32_t>();
        const uint32_t missingMask = result.GetValue(0U), missing = result.GetValue(1U);
        const bool ready = missingMask == 0U;
#else
        ReduceSum(probeSum_, flags_, work_, 1U, flags, 1U);
        Sync<HardEvent::V_S>();
        const bool ready = probeSum_.GetValue(0U) == float(expected);
#endif
        profile_.Elapsed(CombineProfile::ProbeCheckTicks, checkStart);
        profile_.Probe(ready, token, 0xffffffffU, flags, start);
#ifdef DEEPEP_LL_COMBINE_SIMT_FLAG
        profile_.Missing(token, missingMask, missing, result.GetValue(2U), blocks_);
#else
        InspectMissing(token, ready, 0U, k_);
#endif
        return ready;
    }
    __aicore__ inline void InspectMissing(uint32_t token, bool ready, uint32_t firstSlot, uint32_t slots,
                                         uint32_t selectedMask = 0U, bool tokenReady = true)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (!profile_.DetailClock()) return;
        const uint64_t begin = profile_.DetailClock();
        uint32_t mask = 0U, first = 0xffffffffU, count = 0U;
        if (!ready) {
            const uint32_t active = selectedMask ? selectedMask : (enableDedup_ ? masks_.GetValue(token - begin_)
                : (k_ == 32U ? 0xffffffffU : (1U << k_) - 1U));
            for (uint32_t s = 0; s < slots; ++s) {
                if (!((active >> (firstSlot + s)) & 1U)) continue;
                for (uint32_t b = 0; b < blocks_; ++b) {
                    if (flags_.GetValue((s * blocks_ + b) * 8U) != 1.0f) {
                        mask |= 1U << (firstSlot + s);
                        if (first == 0xffffffffU) first = (firstSlot + s) * blocks_ + b;
                        ++count;
                    }
                }
            }
        }
        // A successful single-slot probe does not confirm the whole token.
        // A failed sum with no missing active flag is also not a ready event.
        if (mask || (ready && tokenReady && firstSlot == 0U && slots == k_))
            profile_.Missing(token, mask, first, count, blocks_);
        profile_.Elapsed(CombineProfile::TailInspectTicks, begin);
#endif
    }
#ifdef DEEPEP_LL_COMBINE_PROGRESS_PROBE
    __aicore__ inline bool ProbeProgress(uint32_t token, uint32_t next, uint32_t scan)
    {
        const uint64_t start = profile_.DetailClock();
        const uint32_t mask = enableDedup_ ? masks_.GetValue(token - begin_) : 0U;
        uint32_t readBlocks = 0U;
        for (; next < k_; ++next) {
            if (enableDedup_ && ((mask >> next) & 1U) == 0U) continue;
            const uint64_t loadStart = profile_.DetailClock();
            auto src = resultFlags_[(uint64_t(token) * k_ + next) * packetBytes_ / 4U + 120U];
            // Check every block, including the tail. No flag is cleared until
            // all tokens have arrived, so confirmed contributions stay valid.
            DataCopy(flags_, src, {uint16_t(blocks_), 1U, 15U, 0U});
            Sync<HardEvent::MTE2_V>();
            profile_.Elapsed(CombineProfile::ProbeLoadTicks, loadStart);
            const uint64_t checkStart = profile_.DetailClock();
            ReduceSum(probeSum_, flags_, work_, 1U, blocks_, 1U);
            Sync<HardEvent::V_S>();
            const bool ready = probeSum_.GetValue(0U) == float(blocks_);
            profile_.Elapsed(CombineProfile::ProbeCheckTicks, checkStart);
            readBlocks += blocks_;
            if (!ready) {
                pending_.SetValue(scan, token | (next << 16U));
                profile_.Probe(false, token, next, readBlocks, start);
                InspectMissing(token, false, next, 1U);
                return false;
            }
            profile_.Add(CombineProfile::ProbeConfirmedSlots);
        }
        // The reduction still starts only after all active contributions are
        // ready, and uses the original top-k arithmetic order.
        profile_.Probe(true, token, 0xffffffffU, readBlocks, start);
        profile_.Missing(token, 0U, 0xffffffffU, 0U, blocks_);
        return true;
    }
#endif
    __aicore__ inline void ClearFlags()
    {
        const uint64_t start = profile_.DetailClock();
        const uint32_t flags = k_ * blocks_;
        Duplicate<float>(flags_, 0.0f, flags * 8U);
        // M4 may still own an unconsumed V_MTE3(7) publication for slot 7.
        // Publish zero via Scalar instead: neither wait drains MTE3, and
        // S_MTE3(7) is independent of the inherited payload event directions.
        Sync<HardEvent::V_S>();
        Sync<HardEvent::S_MTE3>();
        for (uint32_t p = begin_; p < begin_ + count_; ++p) {
            auto target = resultFlags_[uint64_t(p) * k_ * packetBytes_ / 4U + 120U];
            DataCopyPad(target, flags_, {uint16_t(flags), 32U, 0U, 480U, 0U});
        }
        profile_.Elapsed(CombineProfile::ClearTicks, start);
    }
    __aicore__ inline void Load(uint32_t token, uint32_t k, uint32_t member)
    {
        const uint64_t start = profile_.DetailClock();
        const uint32_t lane = (inputSequence_ + member) % inputLanes_;
        WaitFlag<HardEvent::V_MTE2>(lane);
        const uint32_t retire = enableDedup_ ? finalLayout_.inputRetireMasks[lane] & ~retiredOutputs_ : 0U;
        if (retire != 0U) {
            for (uint32_t slot = 0; slot < sharedOutputLanes_; ++slot)
                if ((retire >> slot) & 1U) WaitFlag<HardEvent::MTE3_V>(slot);
            // MTE3_V protects Vector, not MTE2. Bridge only first-use reclamation
            // through Vector before this DMA overwrites any inherited output.
            SetFlag<HardEvent::V_MTE2>(lane);
            WaitFlag<HardEvent::V_MTE2>(lane);
            retiredOutputs_ |= retire;
        }
        profile_.Elapsed(CombineProfile::FinalReuseWaitTicks, start);
        profile_.Add(CombineProfile::Contributions);
        const uint64_t issueStart = profile_.DetailClock();
        DataCopy(inputs_[finalLayout_.inputOffsets[lane] / 2U], result_[(uint64_t(token) * k_ + k) * packetBytes_ / 2U],
                 {uint16_t(blocks_), 15U, 1U, 0U});
        SetFlag<HardEvent::MTE2_V>(lane);
        profile_.Elapsed(CombineProfile::FinalInputIssueTicks, issueStart);
    }
#if defined(DEEPEP_LL_COMBINE_EARLY_REDUCE) || defined(DEEPEP_LL_COMBINE_LOCAL_FIRST)
    struct TokenProgress {
        uint32_t token = 0U, nextK = 0U, members = 0U;
    };
    __aicore__ inline bool ProbeContribution(uint32_t token, uint32_t k)
    {
        const uint64_t start = profile_.DetailClock();
        // A contribution is usable only after every 512B block is present,
        // including the final block of a cross-host return packet.
        auto src = resultFlags_[(uint64_t(token) * k_ + k) * packetBytes_ / 4U + 120U];
        DataCopy(flags_, src, {uint16_t(blocks_), 1U, 15U, 0U});
        Sync<HardEvent::MTE2_V>();
        profile_.Elapsed(CombineProfile::ProbeLoadTicks, start);
        const uint64_t checkStart = profile_.DetailClock();
        ReduceSum(probeSum_, flags_, work_, 1U, blocks_, 1U);
        Sync<HardEvent::V_S>();
        const bool ready = probeSum_.GetValue(0U) == float(blocks_);
        profile_.Elapsed(CombineProfile::ProbeCheckTicks, checkStart);
        profile_.Probe(ready, token, k, blocks_, start);
        InspectMissing(token, ready, k, 1U);
        return ready;
    }
    template<class T>
    __aicore__ inline bool AdvanceToken(TokenProgress& state, uint32_t accumulator, uint32_t completed)
    {
        profile_.Add(CombineProfile::EarlyAdvanceCalls);
        const uint32_t mask = enableDedup_ ? masks_.GetValue(state.token - begin_) : 0U;
        auto sum = sum_[accumulator * computeElements_];
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
        const uint32_t scanEnd = 2U * k_;
        const uint32_t localMask = localMasks_.GetValue(state.token - begin_);
#else
        const uint32_t scanEnd = k_;
#endif
        for (; state.nextK < scanEnd; ++state.nextK) {
            const uint32_t k = state.nextK < k_ ? state.nextK : state.nextK - k_;
            if (enableDedup_ && ((mask >> k) & 1U) == 0U) continue;
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
            const bool local = ((localMask >> k) & 1U) != 0U;
            if (local != (state.nextK < k_)) continue;
            const uint64_t probeStart = profile_.DetailClock();
#endif
            const bool ready = ProbeContribution(state.token, k);
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
            profile_.Elapsed(local ? CombineProfile::LocalProbeTicks : CombineProfile::RemoteProbeTicks, probeStart);
            profile_.Add(local ? CombineProfile::LocalProbeCalls : CombineProfile::RemoteProbeCalls);
            if (!ready) profile_.Add(local ? CombineProfile::LocalProbeMisses : CombineProfile::RemoteProbeMisses);
#endif
            if (!ready) {
                PauseAfterFlagMiss(profile_);
                profile_.Add(CombineProfile::EarlyStalls);
                return false;
            }
            // Ascending slots within each class; local-first scans local then
            // remote, while early-reduce keeps the original single pass. The
            // FP32 sum stays in UB across waits; only the final output is cast.
            // Load retains M4 output retirement and input-lane ownership checks.
            Load(state.token, k, 0U);
            const uint32_t lane = inputSequence_ % inputLanes_;
            const uint64_t inputStart = profile_.DetailClock();
            WaitFlag<HardEvent::MTE2_V>(lane);
            profile_.Elapsed(CombineProfile::FinalInputWaitTicks, inputStart);
            const uint64_t computeStart = profile_.DetailClock();
            Cast(cast_, inputs_[finalLayout_.inputOffsets[lane] / 2U].template ReinterpretCast<T>(),
                 RoundMode::CAST_NONE, h_);
            SetFlag<HardEvent::V_MTE2>(lane);
            PipeBarrier<PIPE_V>();
            const float weight = weights_.GetValue((state.token - begin_) * k_ + k);
            if (state.members == 0U) {
                Muls(sum, cast_, weight, h_);
            } else {
                Muls(product_, cast_, weight, h_);
                PipeBarrier<PIPE_V>();
                Add(sum, sum, product_, h_);
            }
            PipeBarrier<PIPE_V>();
            profile_.Elapsed(CombineProfile::FinalComputeIssueTicks, computeStart);
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
            profile_.Elapsed(local ? CombineProfile::LocalReduceTicks : CombineProfile::RemoteReduceTicks, computeStart);
            profile_.Add(local ? CombineProfile::LocalContributions : CombineProfile::RemoteContributions);
#endif
            ++state.members;
            ++inputSequence_;
        }
        // All active contributions have now been observed and consumed.
        profile_.Missing(state.token, 0U, 0xffffffffU, 0U, blocks_);
        const uint32_t outputLane = finalLayout_.outputSlots[completed % outputLanes_];
        const uint64_t outputStart = profile_.DetailClock();
        WaitFlag<HardEvent::MTE3_V>(outputLane);
        profile_.Elapsed(CombineProfile::FinalOutputWaitTicks, outputStart);
        const uint64_t packStart = profile_.DetailClock();
        auto output = outputs_[outputLane * outputBytes_ / 2U];
        Cast(output.template ReinterpretCast<T>(), sum, RoundMode::CAST_RINT, h_);
        SetFlag<HardEvent::V_MTE3>(outputLane);
        WaitFlag<HardEvent::V_MTE3>(outputLane);
        profile_.Elapsed(CombineProfile::FinalOutputPackTicks, packStart);
        const uint64_t storeStart = profile_.DetailClock();
        DataCopyPad(output_[uint64_t(state.token) * h_], output, {1U, h_ * 2U, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_V>(outputLane);
        profile_.Elapsed(CombineProfile::FinalStoreIssueTicks, storeStart);
        profile_.Mark(CombineProfile::FinalLastStore);
        return true;
    }
    __aicore__ inline uint32_t ProcessEarly()
    {
        const uint32_t active = count_ < 2U ? count_ : 2U;
        TokenProgress state[2];
        for (uint32_t i = 0U; i < active; ++i) state[i].token = begin_ + i;
        uint32_t next = active, completed = 0U;
        while (completed < count_) {
            for (uint32_t i = 0U; i < active; ++i) {
                if (state[i].token >= begin_ + count_) continue;
                const bool done = dtype_ == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16
                    ? AdvanceToken<half>(state[i], i, completed)
                    : AdvanceToken<bfloat16_t>(state[i], i, completed);
                if (done) {
                    ++completed;
                    state[i].token = begin_ + next++;
                    state[i].nextK = state[i].members = 0U;
                }
            }
        }
        // Both token states have consumed all their contributions. Clear the
        // same full token ranges as the baseline; no probe reuses flags_ now.
        ClearFlags();
        const uint64_t drainStart = profile_.DetailClock();
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) WaitFlag<HardEvent::V_MTE2>(lane);
        profile_.Elapsed(CombineProfile::FinalInputDrainTicks, drainStart);
        return retiredOutputs_;
    }
#endif
    template <class T>
    __aicore__ inline void Aggregate(uint32_t token, uint32_t mask, uint32_t completed)
    {
        uint32_t nextK = 0U, loaded = 0U;
        for (; nextK < k_ && loaded < inputLanes_; ++nextK)
            if (!enableDedup_ || ((mask >> nextK) & 1U)) Load(token, nextK, loaded++);
        uint32_t member = 0U;
        for (uint32_t k = 0U; k < k_; ++k)
            if (!enableDedup_ || ((mask >> k) & 1U)) {
                const uint32_t lane = (inputSequence_ + member) % inputLanes_;
                const uint64_t inputStart = profile_.DetailClock();
                WaitFlag<HardEvent::MTE2_V>(lane);
                profile_.Elapsed(CombineProfile::FinalInputWaitTicks, inputStart);
                const uint64_t castStart = profile_.DetailClock();
                Cast(cast_, inputs_[finalLayout_.inputOffsets[lane] / 2U].template ReinterpretCast<T>(),
                     RoundMode::CAST_NONE, h_);
                SetFlag<HardEvent::V_MTE2>(lane);
                profile_.Elapsed(CombineProfile::FinalComputeIssueTicks, castStart);
                while (enableDedup_ && nextK < k_ && ((mask >> nextK) & 1U) == 0U) ++nextK;
                if (nextK < k_) Load(token, nextK++, loaded++);
                const uint64_t computeStart = profile_.DetailClock();
                PipeBarrier<PIPE_V>();
                const float weight = weights_.GetValue((token - begin_) * k_ + k);
                if (member == 0U)
                    Muls(sum_, cast_, weight, h_);
                else {
                    Muls(product_, cast_, weight, h_);
                    PipeBarrier<PIPE_V>();
                    Add(sum_, sum_, product_, h_);
                }
                PipeBarrier<PIPE_V>();
                profile_.Elapsed(CombineProfile::FinalComputeIssueTicks, computeStart);
                ++member;
            }
        inputSequence_ += member;
        const uint32_t outputLane = finalLayout_.outputSlots[completed % outputLanes_];
        const uint64_t outputStart = profile_.DetailClock();
        WaitFlag<HardEvent::MTE3_V>(outputLane);
        profile_.Elapsed(CombineProfile::FinalOutputWaitTicks, outputStart);
        const uint64_t packStart = profile_.DetailClock();
        auto output = outputs_[outputLane * outputBytes_ / 2U];
        Cast(output.template ReinterpretCast<T>(), sum_, RoundMode::CAST_RINT, h_);
        SetFlag<HardEvent::V_MTE3>(outputLane);
        WaitFlag<HardEvent::V_MTE3>(outputLane);
        profile_.Elapsed(CombineProfile::FinalOutputPackTicks, packStart);
        const uint64_t storeStart = profile_.DetailClock();
        DataCopyPad(output_[uint64_t(token) * h_], output, {1U, h_ * 2U, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_V>(outputLane);
        profile_.Elapsed(CombineProfile::FinalStoreIssueTicks, storeStart);
        profile_.Mark(CombineProfile::FinalLastStore);
    }

#ifdef DEEPEP_LL_COMBINE_BATCH_FINAL
    __aicore__ inline uint32_t ActiveMask(uint32_t token)
    {
        return enableDedup_ ? masks_.GetValue(token - begin_)
            : (k_ == 32U ? 0xffffffffU : (1U << k_) - 1U);
    }
    __aicore__ inline bool ProbeMask(uint32_t token, uint32_t mask, bool local)
    {
        if (mask == 0U) return true;
        const uint64_t start = profile_.DetailClock();
        auto src = resultFlags_[uint64_t(token) * k_ * packetBytes_ / 4U + 120U];
        DataCopy(flags_, src, {uint16_t(k_ * blocks_), 1U, 15U, 0U});
        Sync<HardEvent::MTE2_V>();
        profile_.Elapsed(CombineProfile::ProbeLoadTicks, start);
        const uint64_t checkStart = profile_.DetailClock();
        // Mask whole slots on Vector, then check every selected 512B block in
        // one reduction and one V_S handoff. Never scalar-read block flags here.
        uint64_t laneMask[2] = {1U, 0U};
        for (uint32_t slot = 0U; slot < k_; ++slot)
            if (((mask >> slot) & 1U) == 0U)
                Duplicate<float>(flags_[slot * blocks_ * 8U], 0.0f, laneMask, uint8_t(blocks_), 1U, 1U);
        PipeBarrier<PIPE_V>();
        ReduceSum(probeSum_, flags_, work_, 1U, k_ * blocks_, 1U);
        Sync<HardEvent::V_S>();
        const bool ready = probeSum_.GetValue(0U) == float(GetBitCount<1>(uint64_t(mask)) * blocks_);
        profile_.Elapsed(CombineProfile::ProbeCheckTicks, checkStart);
        profile_.Probe(ready, token, 0xffffffffU, k_ * blocks_, start, false);
        InspectMissing(token, ready, 0U, k_, mask, false);
        profile_.Add(CombineProfile::BatchGroupCalls);
        profile_.Elapsed(local ? CombineProfile::LocalProbeTicks : CombineProfile::RemoteProbeTicks, start);
        profile_.Add(local ? CombineProfile::LocalProbeCalls : CombineProfile::RemoteProbeCalls);
        if (!ready) profile_.Add(local ? CombineProfile::LocalProbeMisses : CombineProfile::RemoteProbeMisses);
        return ready;
    }
    template<class T>
    __aicore__ inline void ConsumeMask(uint32_t token, uint32_t mask, uint32_t accumulator,
                                      uint32_t& members, bool local)
    {
        if (!mask) return;
        auto sum = sum_[accumulator * computeElements_];
        uint32_t next = 0U, loaded = 0U, consumed = 0U;
        for (; next < k_ && loaded < inputLanes_; ++next)
            if ((mask >> next) & 1U) Load(token, next, loaded++);
        for (uint32_t k = 0U; k < k_; ++k) {
            if (((mask >> k) & 1U) == 0U) continue;
            const uint32_t lane = (inputSequence_ + consumed) % inputLanes_;
            const uint64_t inputStart = profile_.DetailClock();
            WaitFlag<HardEvent::MTE2_V>(lane);
            profile_.Elapsed(CombineProfile::FinalInputWaitTicks, inputStart);
            const uint64_t castStart = profile_.DetailClock();
            Cast(cast_, inputs_[finalLayout_.inputOffsets[lane] / 2U].template ReinterpretCast<T>(),
                 RoundMode::CAST_NONE, h_);
            SetFlag<HardEvent::V_MTE2>(lane);
            profile_.Elapsed(CombineProfile::FinalComputeIssueTicks, castStart);
            // Refill the released input lane before issuing this contribution's
            // multiply/add, as in the baseline multi-lane reduction pipeline.
            while (next < k_ && ((mask >> next) & 1U) == 0U) ++next;
            if (next < k_) Load(token, next++, loaded++);
            const uint64_t computeStart = profile_.DetailClock();
            PipeBarrier<PIPE_V>();
            const float weight = weights_.GetValue((token - begin_) * k_ + k);
            if (members == 0U) Muls(sum, cast_, weight, h_);
            else {
                Muls(product_, cast_, weight, h_);
                PipeBarrier<PIPE_V>();
                Add(sum, sum, product_, h_);
            }
            PipeBarrier<PIPE_V>();
            profile_.Elapsed(CombineProfile::FinalComputeIssueTicks, computeStart);
            profile_.Elapsed(local ? CombineProfile::LocalReduceTicks : CombineProfile::RemoteReduceTicks, computeStart);
            profile_.Add(local ? CombineProfile::LocalContributions : CombineProfile::RemoteContributions);
            ++members;
            ++consumed;
        }
        inputSequence_ += consumed;
    }
    template<class T>
    __aicore__ inline void StoreBatched(uint32_t token, uint32_t accumulator, uint32_t completed)
    {
        const uint32_t lane = finalLayout_.outputSlots[completed % outputLanes_];
        const uint64_t waitStart = profile_.DetailClock();
        WaitFlag<HardEvent::MTE3_V>(lane);
        profile_.Elapsed(CombineProfile::FinalOutputWaitTicks, waitStart);
        const uint64_t packStart = profile_.DetailClock();
        auto output = outputs_[lane * outputBytes_ / 2U];
        Cast(output.template ReinterpretCast<T>(), sum_[accumulator * computeElements_], RoundMode::CAST_RINT, h_);
        SetFlag<HardEvent::V_MTE3>(lane);
        WaitFlag<HardEvent::V_MTE3>(lane);
        profile_.Elapsed(CombineProfile::FinalOutputPackTicks, packStart);
        const uint64_t storeStart = profile_.DetailClock();
        DataCopyPad(output_[uint64_t(token) * h_], output, {1U, h_ * 2U, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE3_V>(lane);
        profile_.Elapsed(CombineProfile::FinalStoreIssueTicks, storeStart);
        profile_.Mark(CombineProfile::FinalLastStore);
    }
    template<class T>
    __aicore__ inline void BatchLoop()
    {
        if (!batchIncremental_) {
            // Many owned tokens: scan all pending tokens, without parking two
            // accumulators on missing remote returns and blocking ready tokens.
            uint32_t remaining = count_, scan = 0U, completed = 0U;
            while (remaining) {
                const uint32_t task = pending_.GetValue(scan), token = task & 0xffffU;
                if (!Probe(token, task >> 16U)) {
                    PauseAfterFlagMiss(profile_);
                    if (++scan == remaining) scan = 0U;
                    continue;
                }
                const uint32_t mask = ActiveMask(token), local = mask & localMasks_.GetValue(token - begin_);
                uint32_t members = 0U;
                ConsumeMask<T>(token, local, 0U, members, true);
                ConsumeMask<T>(token, mask & ~local, 0U, members, false);
                StoreBatched<T>(token, 0U, completed++);
                profile_.Add(CombineProfile::BatchWholeTokens);
                pending_.SetValue(scan, pending_.GetValue(--remaining));
                if (scan == remaining) scan = 0U;
            }
        } else {
            // At most two tokens: retain their FP32 sums while RDMA returns
            // arrive. Each local/remote class has a single vector readiness test.
            uint32_t phase[2] = {}, members[2] = {}, completed = 0U;
            while (completed < count_) {
                for (uint32_t i = 0U; i < count_; ++i) {
                    if (phase[i] == 2U) continue;
                    const uint32_t token = begin_ + i, mask = ActiveMask(token);
                    const uint32_t local = mask & localMasks_.GetValue(i);
                    const bool localPhase = phase[i] == 0U;
                    const uint32_t selected = localPhase ? local : mask & ~local;
                    if (!ProbeMask(token, selected, localPhase)) {
                        PauseAfterFlagMiss(profile_);
                        continue;
                    }
                    // Mark arrival before reduction; computation is not flag wait.
                    if (!localPhase) {
                        profile_.MarkOnce(CombineProfile::FinalFirstReady);
                        profile_.Mark(CombineProfile::FinalLastReady);
                        profile_.Missing(token, 0U, 0xffffffffU, 0U, blocks_);
                    }
                    ConsumeMask<T>(token, selected, i, members[i], localPhase);
                    if (++phase[i] == 2U) StoreBatched<T>(token, i, completed++);
                }
            }
        }
    }
    __aicore__ inline uint32_t ProcessBatched()
    {
        if (dtype_ == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16) BatchLoop<half>();
        else BatchLoop<bfloat16_t>();
        ClearFlags();
        const uint64_t drainStart = profile_.DetailClock();
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) WaitFlag<HardEvent::V_MTE2>(lane);
        profile_.Elapsed(CombineProfile::FinalInputDrainTicks, drainStart);
        return retiredOutputs_;
    }
#endif
    CombineProfile::View profile_;
#ifdef DEEPEP_LL_COMBINE_BATCH_FINAL
    bool batchIncremental_ = false;
    LocalTensor<uint8_t> routeMask_, routeUpper_;
#endif
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
    GM_ADDR sourceIndices_ = nullptr;
    uint32_t rank_ = 0U, expertsPerRank_ = 0U, ranksPerHost_ = 0U;
    LocalTensor<uint32_t> localMasks_;
#endif
    bool tinyProbe_ = false;
    uint32_t tinyPendingMask_ = 0U, tinyActiveSlots_ = 0U;
    bool enableDedup_ = false;
    bool uncached_ = false;
    LocalTensor<uint8_t> scratch_;
    LocalTensor<uint32_t> masks_, pending_;
    LocalTensor<float> weights_, flags_, work_, probeSum_, cast_, product_, sum_;
    LocalTensor<uint16_t> inputs_, outputs_;
    GlobalTensor<uint16_t> result_, output_;
    GlobalTensor<float> resultFlags_;
    uint32_t used_ = 0U, h_ = 0U, k_ = 0U, dtype_ = 0U, begin_ = 0U, count_ = 0U;
    uint32_t blocks_ = 0U, packetBytes_ = 0U, outputBytes_ = 0U;
    EpCombineM5WorkspaceLayout finalLayout_{};
    uint32_t inputLanes_ = 0U, outputLanes_ = 0U, inputSequence_ = 0U;
    uint32_t sharedOutputLanes_ = 0U, retiredOutputs_ = 0U;
#if defined(DEEPEP_LL_COMBINE_EARLY_REDUCE) || defined(DEEPEP_LL_COMBINE_LOCAL_FIRST)
    bool incrementalReduce_ = false;
    uint32_t computeElements_ = 0U;
#endif
};
}  // namespace CombineDedup
#endif
