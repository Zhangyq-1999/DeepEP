#pragma once
#include "kernel_operator.h"
#include "combine_notify_layout.h"

namespace CombineDedup {
#if DEEPEP_LL_COMBINE_NOTIFY > 0
// peerBases is a retired UB scratch containing only same-host mapped addresses.
// The caller must complete MTE3 payload stores BEFORE entering this VF; a SIMT
// thread fence by itself does not order a different execution engine's stores.
__simt_vf__ __launch_bounds__(128) inline void PublishCombineReadyVf(
    __ubuf__ uint64_t* peerBases, uint32_t peers, uint32_t receivers,
    uint32_t rowBytes, uint32_t wordOffset)
{
    for (uint32_t i = threadIdx.x; i < peers * receivers; i += 128U) {
        const uint64_t address = peerBases[i / receivers] + uint64_t(i % receivers) * rowBytes + wordOffset;
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(address), uint32_t(0x3f800000U)); // float 1
    }
    asc_threadfence();
}
#endif
}  // namespace CombineDedup
