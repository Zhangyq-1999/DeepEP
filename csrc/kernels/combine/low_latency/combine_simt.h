#pragma once
#include "kernel_operator.h"

namespace CombineDedup {
// SIMT consumes only MTE-loaded UB snapshots. No SIMT reads of mutable GM,
// no extra workspace, and no change to the order of FP32 accumulation.
__simt_vf__ __launch_bounds__(32) inline void PartialSelectCountVf(
    __ubuf__ int32_t* table, __ubuf__ int32_t* types, __ubuf__ uint32_t* work,
    uint32_t span, uint32_t lower, uint32_t upper, uint32_t world)
{
    const uint32_t lane = threadIdx.x, chunk = (span + 31U) / 32U;
    uint32_t bits = 0U, count = 0U;
    for (uint32_t j = 0; j < chunk; ++j) {
        const uint32_t row = lane * chunk + j;
        if (row >= span || types[row] != 2) continue;
        const uint32_t rank = uint32_t(table[row * 3U]);
        if (rank >= world) bits |= 0x80000000U;
        if (rank >= lower && rank < upper) { bits |= 1U << j; ++count; }
    }
    work[lane] = count;
    work[32U + lane] = bits;
}

__simt_vf__ __launch_bounds__(32) inline void PartialSelectWriteVf(
    __ubuf__ uint32_t* work, __ubuf__ uint32_t* summary, __ubuf__ int32_t* rows,
    uint32_t span, uint32_t begin, uint32_t owner, uint32_t core, uint32_t cores,
    uint32_t outputBegin)
{
    const uint32_t lane = threadIdx.x, chunk = (span + 31U) / 32U;
    uint32_t prefix = 0U;
    for (uint32_t i = 0; i < lane; ++i) prefix += work[i];
    const uint32_t first = (core + cores - owner) % cores;
    uint32_t ordinal = prefix;
    const uint32_t bits = work[32U + lane];
    for (uint32_t j = 0; j < chunk; ++j) {
        if (!(bits & (1U << j))) continue;
        if (ordinal >= first && (ordinal - first) % cores == 0U)
            rows[outputBegin + (ordinal - first) / cores] = int32_t(begin + lane * chunk + j);
        ++ordinal;
    }
    if (lane == 31U) {
        uint32_t invalid = 0U;
        for (uint32_t i = 0; i < 32U; ++i) invalid |= work[32U + i] & 0x80000000U;
        summary[0] = prefix + work[lane];
        summary[1] = invalid;
    }
}

// One lane per contribution, checking EVERY block, including the last one.
// Scratch uses the existing ReduceSum work area (at least 256 B).
__simt_vf__ __launch_bounds__(32) inline void ResultFlagsVf(
    __ubuf__ float* flags, __ubuf__ uint32_t* work, uint32_t k, uint32_t blocks, uint32_t active)
{
    const uint32_t slot = threadIdx.x;
    uint32_t first = 0xffffffffU, missing = 0U;
    if (slot < k && ((active >> slot) & 1U)) {
        for (uint32_t b = 0; b < blocks; ++b) {
            if (flags[(slot * blocks + b) * 8U] != 1.0f) {
                if (first == 0xffffffffU) first = b;
                ++missing;
            }
        }
    }
    work[slot] = first;
    work[32U + slot] = missing;
}

__simt_vf__ __launch_bounds__(32) inline void ResultFlagsSummaryVf(
    __ubuf__ uint32_t* work, __ubuf__ uint32_t* summary, uint32_t k, uint32_t blocks)
{
    if (threadIdx.x != 0U) return;
    uint32_t mask = 0U, first = 0xffffffffU, count = 0U;
    for (uint32_t slot = 0; slot < k; ++slot) {
        count += work[32U + slot];
        if (work[slot] != 0xffffffffU) {
            mask |= 1U << slot;
            if (first == 0xffffffffU) first = slot * blocks + work[slot];
        }
    }
    summary[0] = mask;
    summary[1] = first;
    summary[2] = count;
}
}  // namespace CombineDedup
