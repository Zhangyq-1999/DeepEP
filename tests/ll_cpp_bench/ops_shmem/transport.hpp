#pragma once
#include "kernel_operator.h"
// SHMEM's RDMA backends require cache helpers from the public header first.
#include "shmem.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#ifdef DEEPEP_LL_COMBINE_PROFILE
#include "../../../csrc/kernels/combine/low_latency/combine_profile_device.h"
#endif

namespace opsbench {
// A handle contains a PE number, not an HCCL channel. The upstream batch
// boundary is retained; SHMEM posts each Put and quiets at that boundary.
struct Batch {
    uint32_t peer;
    bool pending = false;
};
class Transport {
    AscendC::LocalTensor<uint8_t> scratch_;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    CombineProfile::View profile_;
#endif

public:
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
    }
    __aicore__ inline void BatchCommit(Batch& batch)
    {
        if (batch.pending) {
#ifdef DEEPEP_LL_COMBINE_PROFILE
            const auto begin = profile_.DetailClock();
#endif
            aclshmemx_roce_qp_quiet(batch.peer, AscendC::GetBlockIdx(), (__ubuf__ uint8_t*)scratch_.GetPhyAddr(), 7U);
#ifdef DEEPEP_LL_COMBINE_PROFILE
            profile_.SampleMax(CombineProfile::RdmaQuietTicks, CombineProfile::RdmaQuietMaxTicks,
                              CombineProfile::RdmaQuietMaxPeer, CombineProfile::RdmaQuietMaxPeer,
                              begin, batch.peer, batch.peer);
            profile_.Add(CombineProfile::RdmaQuietCalls);
#endif
            batch.pending = false;
        }
    }
};
}  // namespace opsbench
