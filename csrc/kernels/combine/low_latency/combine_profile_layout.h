#pragma once
#include <cstdint>

namespace CombineProfile {
// One 512-byte, cacheline-aligned record per AIV. Timestamp fields precede StageRows.
enum Word : uint32_t {
    Entry,
    MagicEnd,
    InitEnd,
    StageBegin,
    StageEnd,
    NotifyBegin,
    NotifyEnd,
    ReadyBegin,
    ReadyEnd,
    PartialBegin,
    PlanBegin,
    PlanEnd,
    PartialRunBegin,
    PartialRunEnd,
    PartialEnd,
    FinalBegin,
    FinalInitEnd,
    FinalEnd,
    DrainEnd,
    DirectBegin,
    DirectPrepared,
    DirectEnd,
    PipeDone,
    FlushDone,
    StageRows,
    DirectRows,
    PartialRows,
    FinalTokens,
    InputLanes,
    OutputLanes,
    DirectScanTicks,
    DirectMetadataTicks,
    DirectInputWaitTicks,
    DirectOutputWaitTicks,
    DirectPackTicks,
    ProbeTicks,
    ProbeCalls,
    ProbeMisses,
    ProbeFlagBlocks,
    FinalInputWaitTicks,
    FinalReuseWaitTicks,
    FinalOutputWaitTicks,
    ClearTicks,
    Contributions,
    Rank,
    Core,
    Tokens,
    World,
    Schema,
    Complete,
    Valid,
    Level,
    Count
};
constexpr uint32_t kWords = 64U;
constexpr uint64_t kSchema = 1U;
static_assert(Count <= kWords && StageRows <= 64U, "profile record bounds");
}  // namespace CombineProfile
