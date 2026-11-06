#pragma once
#include "combine_cache_policy.h"
#include <cstdint>
#ifndef DEEPEP_LL_COMBINE_READY
#define DEEPEP_LL_COMBINE_READY 0
#endif

namespace CombineDedup {
// Match the reference Dispatch's default-L2 control views. Payload and
// data-with-flag receive views are deliberately outside this policy.
constexpr bool kCombineControlL2 = DEEPEP_LL_COMBINE_READY == 19 || DEEPEP_LL_COMBINE_READY == 20 ||
                                  DEEPEP_LL_COMBINE_READY == 23;
// Host build identity: modes 0..3 retain their original two-bit encoding.
// Modes 4..7 use the unused notify=3 tag; ready modes cannot enable notify.
// This keeps the upper 16 bits available for the existing poll-nop value.
// Modes 8..11 additionally use the coalesce-rows bit with coalesce disabled:
// that combination is forbidden for coalescing, and ready excludes coalesce.
constexpr uint32_t CombineReadyOptions(uint32_t mode)
{
    // ready20: control L2; ready21/22: eager-edge return pipelines;
    // ready23: L2 return pipeline that also overlaps the first packet.
    // notify=3/coalesce=1 is another unused tag; do not shift into poll-nop.
    if (mode >= 20U) return (3U << 12U) | (1U << 9U) | ((mode - 20U) << 14U);
    // coalesce=3 is reserved too: ready16..19 use that tag with notify=3.
    // Keep poll-nop in the upper 16 bits; ready and coalesce are exclusive.
    if (mode >= 16U) return (3U << 12U) | (3U << 9U) | ((mode - 16U) << 14U);
    // notify=3 is reserved; use the coalesce bit for modes 12..15 without
    // spilling into poll-nop's upper 16 bits. Ready excludes coalescing.
    if (mode >= 12U) return (3U << 12U) | (1U << 10U) | ((mode - 12U) << 14U);
    if (mode >= 8U) return (3U << 12U) | (1U << 11U) | ((mode - 8U) << 14U);
    return mode >= 4U ? ((3U << 12U) | ((mode - 4U) << 14U)) : (mode << 14U);
}
// Relative to Combine's 1 MiB control base. Start at 1.5 MiB absolute:
// Dispatch's persistent count TX occupies 1280 KiB + up to 40 KiB in each bank.
// Neither it nor the private/per-core ready tables may overwrite these epochs.
constexpr uint64_t kSharedReady = 512U * 1024U;
constexpr uint64_t kSharedAck = kSharedReady + 512U;
constexpr uint64_t kSharedConsumed = kSharedReady + 1024U;
constexpr uint64_t kSharedEpoch = kSharedReady + 4096U;
static_assert(kSharedEpoch + 72U * 32U < 1024U * 1024U, "control region bounds");

}  // namespace CombineDedup
