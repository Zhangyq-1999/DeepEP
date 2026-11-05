#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_COMBINE_READY
#define DEEPEP_LL_COMBINE_READY 0
#endif

namespace CombineDedup {
// Host build identity: modes 0..3 retain their original two-bit encoding.
// Modes 4..7 use the unused notify=3 tag; ready modes cannot enable notify.
// This keeps the upper 16 bits available for the existing poll-nop value.
// Modes 8..11 additionally use the coalesce-rows bit with coalesce disabled:
// that combination is forbidden for coalescing, and ready excludes coalesce.
constexpr uint32_t CombineReadyOptions(uint32_t mode)
{
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
