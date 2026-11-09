#pragma once
#include "kernel_operator.h"
#include "policy.hpp"
// SHMEM's RDMA backends require cache helpers from the public header first.
#include "shmem.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#ifdef DEEPEP_LL_COMBINE_PROFILE
#include "../../../csrc/kernels/combine/low_latency/combine_profile_device.h"
#endif

namespace opsbench {
// A handle contains a PE number, not an HCCL channel. Policy 0 preserves the
// batch boundary; experimental policies retire immutable tiny-BS sources later.
struct Batch {
    uint32_t peer;
    bool pending = false;
};
class Transport {
    AscendC::LocalTensor<uint8_t> scratch_;
#if DEEPEP_LL_OPS_POLICY
    bool defer_ = false;
    uint64_t pendingPeers_ = 0;
    uint32_t peers_[16]{};
    uint32_t peerCount_ = 0;
#endif
#ifdef DEEPEP_LL_COMBINE_PROFILE
    CombineProfile::View profile_;
#endif

public:
#if DEEPEP_LL_OPS_POLICY
    __aicore__ inline void EnableDeferred(bool enabled) { defer_ = enabled; }
    __aicore__ inline bool HasPending() const { return pendingPeers_ != 0; }
    __aicore__ inline void Retire(AscendC::LocalTensor<uint8_t> scratch)
    {
        // The producer TPipe was reset; never reuse its old scratch view.
        scratch_ = scratch;
        for (uint32_t i = 0; i < peerCount_; ++i) {
            Quiet(peers_[i]);
#ifdef DEEPEP_LL_COMBINE_PROFILE
            profile_.Add(CombineProfile::OpsRetiredPeers);
#endif
        }
        pendingPeers_ = 0;
        peerCount_ = 0;
    }
#endif
#ifdef DEEPEP_LL_COMBINE_PROFILE
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* profile) { profile_.Bind(profile); }
#endif
    __aicore__ inline void Init(AscendC::LocalTensor<uint8_t> scratch, uint32_t bytes)
    {
        if (bytes < 128U) AscendC::Trap();
        scratch_ = scratch;
    }
    __aicore__ inline Batch MakeBatchHandle(uint64_t peer, AscendC::LocalTensor<uint8_t>, uint32_t, GM_ADDR, GM_ADDR)
    {
        return {static_cast<uint32_t>(peer), false};
    }
    __aicore__ inline void WriteNbi(Batch& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        // Both pointers are local symmetric addresses. SHMEM translates dst.
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile_.MarkOnce(CombineProfile::RdmaFirstPut);
        const auto begin = profile_.DetailClock();
#endif
        aclshmemx_roce_qp_put_nbi(dst, src, (__ubuf__ uint8_t*)scratch_.GetPhyAddr(), bytes, batch.peer,
                                  AscendC::GetBlockIdx(), 7U);
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile_.Elapsed(CombineProfile::RdmaPutTicks, begin);
        profile_.Mark(CombineProfile::RdmaLastPut);
        profile_.Add(CombineProfile::RdmaPackets);
        profile_.Add(CombineProfile::RdmaBytes, bytes);
#endif
        batch.pending = true;
#if DEEPEP_LL_OPS_POLICY
        if (defer_) {
            const uint64_t bit = uint64_t(1) << batch.peer;
            if (!(pendingPeers_ & bit)) {
                if (peerCount_ == 16U) AscendC::Trap();
                peers_[peerCount_++] = batch.peer;
                pendingPeers_ |= bit;
            }
        }
#endif
    }
    __aicore__ inline void BatchCommit(Batch& batch)
    {
        if (batch.pending) {
#if DEEPEP_LL_OPS_POLICY
            if (defer_) {
                // Put already publishes the WQE/DB. Nothing is held in a batch
                // UB buffer, and each GM payload stays immutable until Retire.
#ifdef DEEPEP_LL_COMBINE_PROFILE
                profile_.Add(CombineProfile::OpsDeferredCommits);
#endif
                batch.pending = false;
                return;
            }
#endif
            Quiet(batch.peer);
            batch.pending = false;
        }
    }
private:
    __aicore__ inline void Quiet(uint32_t peer)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
            const auto begin = profile_.DetailClock();
#endif
            aclshmemx_roce_qp_quiet(peer, AscendC::GetBlockIdx(), (__ubuf__ uint8_t*)scratch_.GetPhyAddr(), 7U);
#ifdef DEEPEP_LL_COMBINE_PROFILE
            profile_.SampleMax(CombineProfile::RdmaQuietTicks, CombineProfile::RdmaQuietMaxTicks,
                              CombineProfile::RdmaQuietMaxPeer, CombineProfile::RdmaQuietMaxPeer,
                              begin, peer, peer);
            profile_.Add(CombineProfile::RdmaQuietCalls);
#endif
    }
};
}  // namespace opsbench
