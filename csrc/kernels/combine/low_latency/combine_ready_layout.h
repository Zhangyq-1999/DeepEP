#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_COMBINE_READY
#define DEEPEP_LL_COMBINE_READY 0
#endif

namespace CombineDedup {
// Relative to Combine's 1 MiB control base. Start at 1.5 MiB absolute:
// Dispatch's persistent count TX occupies 1280 KiB + up to 40 KiB in each bank.
// Neither it nor the private/per-core ready tables may overwrite these epochs.
constexpr uint64_t kSharedReady = 512U * 1024U;
constexpr uint64_t kSharedAck = kSharedReady + 512U;
constexpr uint64_t kSharedConsumed = kSharedReady + 1024U;
constexpr uint64_t kSharedEpoch = kSharedReady + 4096U;
static_assert(kSharedEpoch + 72U * 32U < 1024U * 1024U, "control region bounds");

}  // namespace CombineDedup
