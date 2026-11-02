#pragma once
#include "../../common/ep_server_order.h"

namespace CombineDedup {
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define COMBINE_RETURN_INLINE __aicore__ inline
#else
#define COMBINE_RETURN_INLINE inline
#endif
// Move the local physical host to the end, preserving the relative order of
// all remote hosts. G2/G4 are single-host paths and retain their original order.
// Task ownership follows this traversal, so an enabled build also redistributes
// individual tasks among the same set of per-core QPs.
template<uint32_t Group>
COMBINE_RETURN_INLINE uint32_t CombineReturnServer(uint32_t servers, uint32_t own, uint32_t ordinal)
{
#ifdef DEEPEP_LL_COMBINE_REMOTE_FIRST
    if constexpr (Group == 16U) {
        if (servers > 1U) {
            const uint32_t split = servers / 2U;
            const uint32_t ownLength = own < split ? split : servers - split;
            const uint32_t localOrdinal = servers < 16U ? ownLength - 1U : 2U * (ownLength - 1U);
            ordinal = ordinal == servers - 1U ? localOrdinal : ordinal + uint32_t(ordinal >= localOrdinal);
        }
    }
#endif
    return EpServerOrder::ServerDedupHalfOrder(servers, own, ordinal);
}

COMBINE_RETURN_INLINE constexpr bool CombineDeferReturn(uint32_t outputLanes)
{
#ifdef DEEPEP_LL_COMBINE_DEFER_RETURN
    return outputLanes > 1U;
#else
    return false;
#endif
}
#undef COMBINE_RETURN_INLINE
}  // namespace CombineDedup
