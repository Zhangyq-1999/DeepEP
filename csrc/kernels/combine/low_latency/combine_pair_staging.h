#pragma once
#include "../../dispatch/low_latency/device/ll_transport.h"
#include "combine_profile_device.h"

namespace CombineDedup {
// Two independent packets, two WQEs. Only the local MTE3 completion is batched.
// M4 may defer the first packet only when a second task is already runnable and
// owns another output lane. DIRECT retains immediate submission through the
// ordinary SendReturnPacket wrapper. No source slot is reused before Quiet.
class PairStagingBatch {
    static constexpr uint32_t kPeerWords = (DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE + 31U) / 32U;
    struct Packet {
        uint32_t peer;
        uint64_t offset;
        GM_ADDR stage;
        AscendC::LocalTensor<uint8_t> scratch;
    } packets_[2];
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    GM_ADDR stageBase_ = nullptr;
    uint32_t bytes_ = 0U, capacity_ = 0U, count_ = 0U, staged_ = 0U;
    uint32_t peerMask_[kPeerWords]{};
    CombineProfile::View profile_;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    uint64_t queued_ = 0U;
#endif

    __aicore__ inline void Retire(AscendC::LocalTensor<uint8_t> scratch)
    {
        if (!count_) return;
        for (uint32_t word = 0; word < kPeerWords; ++word) {
            const uint32_t mask = peerMask_[word];
            for (uint32_t bit = 0; bit < 32U; ++bit) if (mask & (1U << bit)) {
                const uint32_t peer = word * 32U + bit;
                const uint64_t begin = profile_.DetailClock();
                LlTransport::Quiet(peer, scratch);
                profile_.SampleMax(CombineProfile::RdmaQuietTicks, CombineProfile::RdmaQuietMaxTicks,
                                   CombineProfile::RdmaQuietMaxPeer, CombineProfile::RdmaQuietMaxPeer,
                                   begin, peer, peer);
                profile_.Add(CombineProfile::RdmaQuietCalls);
            }
            peerMask_[word] = 0U;
        }
        count_ = 0U;
    }
public:
    __aicore__ inline void SetCombineProfile(CombineProfile::DeviceProfile* p) { profile_.Bind(p); }
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, uint32_t bytes)
    {
        if (count_ || staged_ || !bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) {
            AscendC::Trap(); return;
        }
        args_ = args; bytes_ = bytes; capacity_ = LlRdma::kCoreBytes / bytes;
        if (uint32_t(args->rankSize) > DispatchDedup::kLlPhysicalRanksPerHost)
            stageBase_ = LlTransport::Stage(args, capacity_ * bytes);
    }
    __aicore__ inline bool Pending() const { return count_ != 0U; }
    __aicore__ inline bool HasStaged() const { return staged_ != 0U; }
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset,
                                   AscendC::LocalTensor<uint8_t> packet, bool defer = false)
    {
        LlTransport::Check(args_, peer, offset, bytes_);
        if (staged_ == 2U || count_ == capacity_) { AscendC::Trap(); return; }
        auto& item = packets_[staged_];
        item.peer = peer; item.offset = offset; item.stage = stageBase_ + uint64_t(count_) * bytes_;
        item.scratch = packet;
        const uint64_t begin = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstStage);
        AscendC::DataCopy(LlTransport::Tensor(item.stage, true), packet, bytes_);
        profile_.Elapsed(CombineProfile::RdmaStageIssueTicks, begin);
        profile_.Mark(CombineProfile::RdmaLastStage);
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (!staged_) queued_ = profile_.DetailClock();
#endif
        ++staged_; ++count_;
        peerMask_[peer / 32U] |= 1U << (peer % 32U);
        const bool hold = defer && staged_ < 2U && count_ < capacity_;
        profile_.Add(CombineProfile::RdmaDeferredPackets, hold ? 1U : 0U);
        if (!hold) SubmitPending();
    }
    __aicore__ inline void SubmitPending()
    {
        if (!staged_) return;
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile_.Elapsed(CombineProfile::RdmaDeferGapTicks, queued_);
#endif
        // One ticket follows every queued DMA. Consume it before the SDK may
        // borrow event 7 and the first 128 bytes of either output lane.
        const uint64_t issue = profile_.DetailClock();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(7U);
        profile_.Elapsed(CombineProfile::RdmaStageIssueTicks, issue);
        const uint64_t wait = profile_.DetailClock();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(7U);
        profile_.SampleMax(CombineProfile::RdmaStageWaitTicks, CombineProfile::RdmaStageWaitMaxTicks,
                           CombineProfile::RdmaStageWaitMaxPeer, CombineProfile::RdmaStageWaitMaxPeer,
                           wait, packets_[staged_ - 1U].peer, packets_[staged_ - 1U].peer);
        profile_.Add(CombineProfile::RdmaStageWaitCalls);
        profile_.Add(CombineProfile::RdmaStagePairGroups, staged_ == 2U ? 1U : 0U);
        for (uint32_t i = 0; i < staged_; ++i) {
            auto& item = packets_[i];
            const uint64_t begin = profile_.DetailClock();
            profile_.MarkOnce(CombineProfile::RdmaFirstPut);
            LlTransport::Put(args_, item.peer, LlTransport::RemoteAddress(args_, item.peer, item.offset),
                             item.stage, bytes_, item.scratch);
            profile_.Elapsed(CombineProfile::RdmaPutTicks, begin);
            profile_.Add(CombineProfile::RdmaPackets);
            profile_.Mark(CombineProfile::RdmaLastPut);
        }
        auto scratch = packets_[staged_ - 1U].scratch;
        staged_ = 0U;
        if (count_ == capacity_) Retire(scratch);
    }
    __aicore__ inline void Flush(AscendC::LocalTensor<uint8_t> scratch)
    {
        SubmitPending();
        Retire(scratch);
    }
};
}
