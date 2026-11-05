#pragma once
#include "../../dispatch/low_latency/device/ll_transport.h"
#include "combine_profile_device.h"
#if DEEPEP_LL_COMBINE_READY == 9
#include "combine_wqe_prepare.h"
#endif

namespace CombineDedup {
// Immediate-submit experiments: preserve the original full GM staging batch
// and one quiet per used peer before any GM slot is recycled. M5 may overlap
// the final outstanding NIC reads; Flush still retires them at kernel exit.
class TransferBatch {
    static constexpr uint32_t kWords = (DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE + 31U) / 32U;
    const __gm__ LlTransport::Args* args_ = nullptr;
    uint32_t bytes_ = 0U, count_ = 0U, capacity_ = 0U;
    uint32_t peers_[kWords]{};
    GM_ADDR stage_ = nullptr;
    CombineProfile::View profile_;
#if DEEPEP_LL_COMBINE_READY == 9
    AscendC::LocalTensor<uint8_t> privateUb_;
    bool partial_ = false;
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
        LlTransport::Check(args_, peer, offset, bytes_);
        return stage_ + uint64_t(count_) * bytes_;
    }
    __aicore__ inline void Submit(uint32_t peer, uint64_t offset, GM_ADDR src,
                                  AscendC::LocalTensor<uint8_t> scratch)
    {
        using namespace AscendC;
        SetFlag<HardEvent::MTE3_S>(7U);
        profile_.Mark(CombineProfile::RdmaLastStage);
#if DEEPEP_LL_COMBINE_READY == 9
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
        WaitFlag<HardEvent::MTE3_S>(7U);
        profile_.Add(CombineProfile::RdmaStageWaitCalls);
        profile_.SampleMax(CombineProfile::RdmaStageWaitTicks, CombineProfile::RdmaStageWaitMaxTicks,
            CombineProfile::RdmaStageWaitMaxPeer, CombineProfile::RdmaStageWaitMaxPeer, wait, peer, peer);
        const uint64_t put = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstPut);
#if DEEPEP_LL_COMBINE_READY == 9
        if (prepared) write.Publish();
        else
#endif
            LlTransport::Put(args_, peer, LlTransport::RemoteAddress(args_, peer, offset), src, bytes_, scratch);
        profile_.Elapsed(CombineProfile::RdmaPutTicks, put);
        profile_.Add(CombineProfile::RdmaPackets);
        profile_.Mark(CombineProfile::RdmaLastPut);
        peers_[peer / 32U] |= 1U << (peer % 32U);
        if (++count_ == capacity_) Retire(scratch);
    }
public:
    __aicore__ inline void SetCombineProfile(CombineProfile::DeviceProfile* p) { profile_.Bind(p); }
    __aicore__ inline void Init(const __gm__ LlTransport::Args* args, uint32_t bytes)
    {
        if (count_ || !bytes || bytes % 512U || bytes > LlRdma::kCoreBytes) { AscendC::Trap(); return; }
        args_ = args; bytes_ = bytes; capacity_ = LlRdma::kCoreBytes / bytes;
        if (uint32_t(args->rankSize) > DispatchDedup::kLlPhysicalRanksPerHost)
            stage_ = LlTransport::Stage(args, capacity_ * bytes);
    }
#if DEEPEP_LL_COMBINE_READY == 9
    __aicore__ inline void SetPartialScratch(AscendC::LocalTensor<uint8_t> scratch)
    { privateUb_ = scratch; partial_ = true; }
#endif
    __aicore__ inline bool Pending() const { return count_ != 0U; }
    __aicore__ inline bool HasStaged() const { return false; }
    __aicore__ inline void SubmitPending() {}
    __aicore__ inline void Enqueue(uint32_t peer, uint64_t offset, AscendC::LocalTensor<uint8_t> packet, bool = false)
    {
        const auto dst = Next(peer, offset);
        const uint64_t begin = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstStage);
        AscendC::DataCopy(LlTransport::Tensor(dst, true), packet, bytes_);
        profile_.Elapsed(CombineProfile::RdmaStageIssueTicks, begin);
        Submit(peer, offset, dst, packet);
    }
    __aicore__ inline void EnqueueLinear(uint32_t peer, uint64_t offset, AscendC::LocalTensor<uint16_t> payload,
                                        AscendC::LocalTensor<uint8_t> packet, uint32_t dataBytes)
    {
        using namespace AscendC;
        const auto dst = Next(peer, offset);
        auto gm = LlTransport::Tensor(dst, true);
        const uint32_t full = dataBytes / 480U, tail = dataBytes % 480U, blocks = bytes_ / 512U;
        const uint64_t begin = profile_.DetailClock();
        profile_.MarkOnce(CombineProfile::RdmaFirstStage);
        auto src = payload.ReinterpretCast<uint8_t>();
        if (full) DataCopyPad(gm, src, {uint16_t(full), 480U, 0U, 32U, 0U});
        if (tail) DataCopyPad(gm[full * 512U], src[full * 480U], {1U, tail, 0U, 0U, 0U});
        // Reuse the original immutable 32B flags in each output slot. UB stride
        // is measured in 32B units; GM stride is bytes. No extra Vector pack.
        DataCopyPad(gm[480U], packet[480U], {uint16_t(blocks), 32U, 15U, 480U, 0U});
        profile_.Elapsed(CombineProfile::RdmaStageIssueTicks, begin);
        profile_.Add(CombineProfile::MtePackPackets);
        Submit(peer, offset, dst, packet); // retires MTE3 before low_ can be overwritten
    }
    __aicore__ inline void Flush(AscendC::LocalTensor<uint8_t> scratch) { Retire(scratch); }
};
}
