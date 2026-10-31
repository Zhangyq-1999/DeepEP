// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>
namespace deepep {
struct EpServerDedupCombineKernelArgs {
    uint8_t *commArgs, *expertOut, *destinationIndex, *sendCounts, *expertScales;
    uint8_t *tokenType, *relayReadIndex, *sourceMask, *yOut, *tilingData;
    uint8_t* stageTimer;
    uint32_t timerMode;
    uint32_t coreNum;
};
static_assert(std::is_standard_layout<EpServerDedupCombineKernelArgs>::value, "combine launch ABI");
static_assert(sizeof(EpServerDedupCombineKernelArgs) == 96, "combine launch ABI");
static_assert(offsetof(EpServerDedupCombineKernelArgs, stageTimer) == 80, "timer address");
static_assert(offsetof(EpServerDedupCombineKernelArgs, timerMode) == 88, "timer mode");
static_assert(offsetof(EpServerDedupCombineKernelArgs, coreNum) == 92, "active cores");
}  // namespace deepep
extern "C" void combine_server_dedup_kernel_do(void* stream, const deepep::EpServerDedupCombineKernelArgs* args);
extern "C" void combine_clear_result_flags_kernel_do(void* stream, uint8_t* local, uint64_t blocks, uint32_t cores);
extern "C" uint32_t deepep_ll_combine_small_batch_enabled();
