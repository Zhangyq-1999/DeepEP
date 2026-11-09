#pragma once
#include <cstdint>

// The experimental build changes only the original token's top-k expert IDs.
// Dispatch-produced counts, destination/relay tables and masks remain int32.
#ifndef DEEPEP_LL_COMBINE_SOURCE_INDEX_BITS
#define DEEPEP_LL_COMBINE_SOURCE_INDEX_BITS 32
#endif
#if DEEPEP_LL_COMBINE_SOURCE_INDEX_BITS != 32 && DEEPEP_LL_COMBINE_SOURCE_INDEX_BITS != 64
#error "Combine source indices must be int32 or int64"
#endif
namespace CombineDedup {
#if DEEPEP_LL_COMBINE_SOURCE_INDEX_BITS == 64
using SourceIndex = int64_t;
#else
using SourceIndex = int32_t;
#endif
constexpr uint32_t kSourceIndexBits = DEEPEP_LL_COMBINE_SOURCE_INDEX_BITS;
constexpr uint32_t kSourceIndexBytes = sizeof(SourceIndex);
#ifdef DEEPEP_LL_COMBINE_LOCAL_FIRST
constexpr bool kSourceIndicesRead = true;
#else
constexpr bool kSourceIndicesRead = false;
#endif
}
