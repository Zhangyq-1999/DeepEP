#pragma once
#include <cstdint>
#ifndef DEEPEP_LL_COMBINE_DATA_L2
#define DEEPEP_LL_COMBINE_DATA_L2 0
#endif
namespace CombineDedup {
// Independent from ready/scheduling options: 1=M3 stage, 2=RDMA TX stage,
// 4=M5 result (payload AND flags). Use NORMAL, never persistent caching.
constexpr uint32_t kCombineDataL2 = DEEPEP_LL_COMBINE_DATA_L2;
static_assert(kCombineDataL2 <= 7U, "invalid Combine data L2 mask");
constexpr bool kCombineStageL2 = (kCombineDataL2 & 1U) != 0U;
constexpr bool kCombineTxL2 = (kCombineDataL2 & 2U) != 0U;
constexpr bool kCombineResultL2 = (kCombineDataL2 & 4U) != 0U;
}  // namespace CombineDedup
