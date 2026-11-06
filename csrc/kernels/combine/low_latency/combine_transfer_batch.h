#pragma once
#include "combine_cache_policy.h"
#include "../../dispatch/low_latency/device/ll_transport.h"
#include "combine_profile_device.h"
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
#include "combine_wqe_prepare.h"
#endif

namespace CombineDedup {
// Preserve the original full GM staging batch
// and one quiet per used peer before any GM slot is recycled. M5 may overlap
// the final outstanding NIC reads; Flush still retires them at kernel exit.
class TransferBatch {
    static constexpr uint32_t kWords = (DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE + 31U) / 32U;
    const __gm__ LlTransport::Args* args_ = nullptr;
    uint32_t bytes_ = 0U, count_ = 0U, capacity_ = 0U;
    uint32_t peers_[kWords]{};
    GM_ADDR stage_ = nullptr;
    CombineProfile::View profile_;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    bool splitStageWait_ = false;
#endif
    __aicore__ inline void MarkStagePrefix()
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        // Event 6 captures the already queued MTE3 work/dependencies BEFORE
        // this payload DMA. Do not wait here: retain overlap with WQE build.
        // Event 7 remains the original payload-completion/SDK event.
        splitStageWait_ = profile_.DetailClock() != 0U;
        if (splitStageWait_) AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(6U);
#endif
    }
    __aicore__ inline void WaitStage()
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (splitStageWait_) {
            const uint64_t begin = profile_.DetailClock();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(6U);
            const uint64_t prefixEnd = profile_.DetailClock();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(7U);
            const uint64_t copyEnd = profile_.DetailClock();
            profile_.Add(CombineProfile::RdmaPrefixWaitTicks, prefixEnd - begin);
            profile_.Add(CombineProfile::RdmaCopyWaitTicks, copyEnd - prefixEnd);
            profile_.Add(CombineProfile::RdmaWaitSplitCalls);
            splitStageWait_ = false;
            return;
        }
#endif
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(7U);
    }
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
    AscendC::LocalTensor<uint8_t> privateUb_;
    bool partial_ = false;
#endif
#if DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
    PreparedWrite pendingWrite_;
    AscendC::LocalTensor<uint8_t> pendingScratch_;
    GM_ADDR pendingStage_ = nullptr;
    uint64_t pendingOffset_ = 0U;
    uint32_t pendingPeer_ = 0U;
    bool staged_ = false, prepared_ = false;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    uint64_t queued_ = 0U;
    bool deferred_ = false;
#endif
#endif
    __aicore__ inline void Retire(AscendC::LocalTensor<uint8_t> scratch)
    {
        if (!count_) return;
        for (uint32_t word = 0U; word < kWords; ++word) {
            const uint32_t mask = peers_[word];
            if (!mask) continue;
            for (uint32_t bit = 0U; bit < 32U; ++bit) if (mask & (1U << bit)) {
                const uint32_t peer = word * 32U + bit;
                const uint64_t begin = profile_.DetailClock();
                LlTransport::Quiet(peer, scratch);
                profile_.SampleMax(CombineProfile::RdmaQuietTicks, CombineProfile::RdmaQuietMaxTicks,
                    CombineProfile::RdmaQuietMaxPeer, CombineProfile::RdmaQuietMaxPeer, begin, peer, peer);
                profile_.Add(CombineProfile::RdmaQuietCalls);
            }
            peers_[word] = 0U;
        }
        count_ = 0U;
    }
    __aicore__ inline GM_ADDR Next(uint32_t peer, uint64_t offset)
    {
#if DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
        if (staged_) { AscendC::Trap(); return nullptr; }
#endif
        LlTransport::Check(args_, peer, offset, bytes_);
        return stage_ + uint64_t(count_) * bytes_;
    }
    __aicore__ inline void Submit(uint32_t peer, uint64_t offset, GM_ADDR src,
                                  AscendC::LocalTensor<uint8_t> scratch, bool defer = false)
    {
        using namespace AscendC;
        SetFlag<HardEvent::MTE3_S>(7U);
        profile_.Mark(CombineProfile::RdmaLastStage);
#if DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
        pendingPeer_ = peer; pendingOffset_ = offset; pendingStage_ = src;
        pendingScratch_ = scratch; prepared_ = false; staged_ = true;
        if (partial_) {
            const uint64_t begin = profile_.DetailClock();
            prepared_ = pendingWrite_.Prepare(peer, LlTransport::RemoteAddress(args_, peer, offset),
                                              src, bytes_, privateUb_);
            profile_.Elapsed(CombineProfile::WqePrepareTicks, begin);
            profile_.Add(prepared_ ? CombineProfile::WqePreparedPackets : CombineProfile::WqeFallbackPackets);
        }
        // Only an already-selected following M4 task may extend this lifetime.
        // DIRECT, one-lane and last-ready-task paths still submit immediately.
        const bool park = partial_ && defer;
        profile_.Add(CombineProfile::RdmaDeferredPackets, park ? 1U : 0U);
#ifdef DEEPEP_LL_COMBINE_PROFILE
        queued_ = profile_.DetailClock(); deferred_ = park;
#endif
        if (!park) SubmitPending();
#else
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 12
        PreparedWrite write;
        bool prepared = false;
        if (partial_) {
            const uint64_t begin = profile_.DetailClock();
            prepared = write.Prepare(peer, LlTransport::RemoteAddress(args_, peer, offset), src, bytes_, privateUb_);
            profile_.Elapsed(CombineProfile::WqePrepareTicks, begin);
            profile_.Add(prepared ? CombineProfile::WqePreparedPackets : CombineProfile::WqeFallbackPackets);
        }
#endif
        const uint64_t wait = profile_.DetailClock();
        WaitStage();
        profile_.Add(CombineProfile::RdmaStageWaitCalls);
        profile_.SampleMax(CombineProfile::RdmaStageWaitTicks, CombineProfile::RdmaStageWaitMaxTicks,
            CombineProfile::RdmaStageWaitMaxPeer, CombineProfile::RdmaStageWaitMaxPeer, wait, peer, peer);
        const uint64_t put = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstPut);
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 12
        if (prepared) write.Publish();
        else
#endif
            LlTransport::Put(args_, peer, LlTransport::RemoteAddress(args_, peer, offset), src, bytes_, scratch);
        profile_.Elapsed(CombineProfile::RdmaPutTicks, put);
        profile_.Add(CombineProfile::RdmaPackets);
        profile_.Mark(CombineProfile::RdmaLastPut);
        peers_[peer / 32U] |= 1U << (peer % 32U);
        if (++count_ == capacity_) Retire(scratch);
#endif
    }
public:
    __aicore__ inline void SetCombineProfile(CombineProfile::DeviceProfile* p) { profile_.Bind(p); }
    __aicore__ inline void Init(const __gm__ LlTransport::Args* args, uint32_t bytes)
    {
        if (Pending() || !bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) { AscendC::Trap(); return; }
        args_ = args; bytes_ = bytes; capacity_ = LlRdma::kCoreBytes / bytes;
        if (uint32_t(args->rankSize) > DispatchDedup::kLlPhysicalRanksPerHost)
            stage_ = LlTransport::Stage(args, capacity_ * bytes);
    }
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
    __aicore__ inline void SetPartialScratch(AscendC::LocalTensor<uint8_t> scratch)
    { privateUb_ = scratch; partial_ = true; }
#endif
    __aicore__ inline bool Pending() const { return count_ != 0U || HasStaged(); }
#if DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
    __aicore__ inline bool HasStaged() const { return staged_; }
    __aicore__ inline void SubmitPending()
    {
        if (!staged_) return;
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (deferred_) profile_.Elapsed(CombineProfile::RdmaDeferGapTicks, queued_);
#endif
        const uint64_t wait = profile_.DetailClock();
        WaitStage();
        profile_.Add(CombineProfile::RdmaStageWaitCalls);
        profile_.SampleMax(CombineProfile::RdmaStageWaitTicks, CombineProfile::RdmaStageWaitMaxTicks,
            CombineProfile::RdmaStageWaitMaxPeer, CombineProfile::RdmaStageWaitMaxPeer,
            wait, pendingPeer_, pendingPeer_);
        const uint64_t put = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstPut);
        if (prepared_) pendingWrite_.Publish();
        else LlTransport::Put(args_, pendingPeer_, LlTransport::RemoteAddress(args_, pendingPeer_, pendingOffset_),
                              pendingStage_, bytes_, pendingScratch_);
        profile_.Elapsed(CombineProfile::RdmaPutTicks, put);
        profile_.Add(CombineProfile::RdmaPackets);
        profile_.Mark(CombineProfile::RdmaLastPut);
        peers_[pendingPeer_ / 32U] |= 1U << (pendingPeer_ % 32U);
        staged_ = false;
        if (++count_ == capacity_) Retire(pendingScratch_);
    }
#else
    __aicore__ inline bool HasStaged() const { return false; }
    __aicore__ inline void SubmitPending() {}
#endif
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, AscendC::LocalTensor<uint8_t> packet, bool defer = false)
    {
        const auto dst = Next(peer, offset);
        const uint64_t begin = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstStage);
        MarkStagePrefix();
        AscendC::DataCopy(LlTransport::Tensor(dst, !kCombineTxL2), packet, bytes_);
        profile_.Elapsed(CombineProfile::RdmaStageIssueTicks, begin);
        Submit(peer, offset, dst, packet, defer);
    }
    __aicore__ inline void EnqueueLinear(uint32_t peer, uint64_t offset, AscendC::LocalTensor<uint16_t> payload,
                                        AscendC::LocalTensor<uint8_t> packet, uint32_t dataBytes)
    {
        using namespace AscendC;
        const auto dst = Next(peer, offset);
        auto gm = LlTransport::Tensor(dst, !kCombineTxL2);
        const uint32_t full = dataBytes / 480U, tail = dataBytes % 480U, blocks = bytes_ / 512U;
        const uint64_t begin = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstStage);
        auto src = payload.ReinterpretCast<uint8_t>();
        MarkStagePrefix();
        if (full) DataCopyPad(gm, src, {uint16_t(full), 480U, 0U, 32U, 0U});
        if (tail) DataCopyPad(gm[full * 512U], src[full * 480U], {1U, tail, 0U, 0U, 0U});
        // Reuse the original immutable 32B flags in each output slot. UB stride
        // is measured in 32B units; GM stride is bytes. No extra Vector pack.
        DataCopyPad(gm[480U], packet[480U], {uint16_t(blocks), 32U, 15U, 480U, 0U});
        profile_.Elapsed(CombineProfile::RdmaStageIssueTicks, begin);
        profile_.Add(CombineProfile::MtePackPackets);
        Submit(peer, offset, dst, packet); // retires MTE3 before low_ can be overwritten
    }
    __aicore__ inline void Flush(AscendC::LocalTensor<uint8_t> scratch) { SubmitPending(); Retire(scratch); }
};
}
