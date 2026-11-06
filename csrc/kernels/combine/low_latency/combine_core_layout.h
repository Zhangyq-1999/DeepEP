#pragma once
#include <cstdint>
#include "combine_ready_layout.h"

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define COMBINE_CORE_CONSTEXPR __aicore__ inline constexpr
#else
#define COMBINE_CORE_CONSTEXPR constexpr
#endif

namespace CombineDedup {
// ready18/19 interleave input6's physical placement. Logical IDs keep
// existing task ownership and control-table rows; SDK QPs, transfer staging,
// magic rotation and profile records continue to use physical GetBlockIdx().
constexpr bool kCombineInterleavedCores = DEEPEP_LL_COMBINE_READY == 18 || DEEPEP_LL_COMBINE_READY == 19;

COMBINE_CORE_CONSTEXPR uint32_t CombineLogicalCore(uint32_t physical, uint32_t cores)
{
    if constexpr (kCombineInterleavedCores)
        return physical / 2U + ((physical & 1U) ? 0U : cores / 2U);
    return physical;
}

COMBINE_CORE_CONSTEXPR uint32_t CombinePhysicalCore(uint32_t logical, uint32_t cores)
{
    if constexpr (kCombineInterleavedCores)
        return logical < cores / 2U ? 2U * logical + 1U : 2U * (logical - cores / 2U);
    return logical;
}

COMBINE_CORE_CONSTEXPR bool CombineIsReceiveCore(uint32_t physical, uint32_t cores, uint32_t recv)
{
    return CombineLogicalCore(physical, cores) >= cores - recv;
}

COMBINE_CORE_CONSTEXPR uint32_t CombineReceiveCoreIndex(uint32_t physical, uint32_t cores, uint32_t recv)
{
    return CombineLogicalCore(physical, cores) - (cores - recv);
}
}  // namespace CombineDedup
#undef COMBINE_CORE_CONSTEXPR
