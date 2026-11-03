#pragma once
#include "combine_profile_device.h"

#ifndef DEEPEP_LL_COMBINE_POLL_NOP
#define DEEPEP_LL_COMBINE_POLL_NOP 0
#endif
static_assert(DEEPEP_LL_COMBINE_POLL_NOP >= 0 && DEEPEP_LL_COMBINE_POLL_NOP <= 65535,
              "poll Nop count must fit the exported configuration");

namespace CombineDedup {
// First probe is immediate. Call only after a failed snapshot, before retrying
// or rotating to another token. Keep the original flag reads and ready checks.
__aicore__ inline void PauseAfterFlagMiss(CombineProfile::View& profile)
{
#if DEEPEP_LL_COMBINE_POLL_NOP > 0
    const uint64_t begin = profile.DetailClock();
    // Ascend C's implementation also brackets Nop with PIPE_ALL barriers.
    // Record the whole call, not an assumed fixed conversion to microseconds.
    AscendC::Nop<DEEPEP_LL_COMBINE_POLL_NOP>();
    profile.Elapsed(CombineProfile::PollPauseTicks, begin);
    profile.Add(CombineProfile::PollPauseCalls);
#endif
}
}  // namespace CombineDedup
