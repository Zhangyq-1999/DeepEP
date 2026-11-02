// Copyright (c) 2026, Lu Lu
#pragma once
#include <cstdint>

namespace DispatchDedup {
// Internal launch selection, independent of the shared-memory communication ABI.
// Numeric format values are checked against the public enums by the Host binding.
constexpr uint8_t kLlDispatchFp16 = 3;
constexpr uint8_t kLlDispatchBf16 = 11;
constexpr uint8_t kLlDispatchE4M3 = 15;
constexpr uint8_t kLlDispatchE5M2 = 16;
constexpr uint8_t kLlDispatchPlain16 = 0;
constexpr uint8_t kLlDispatchQuantize16 = 1;
constexpr uint8_t kLlDispatchPrequantized = 2;
constexpr uint32_t kLlDispatchInvalidVariant = 0xffffffffU;
constexpr uint32_t kLlDispatchVariantCount = 16U;

// mode: prequant E4M3, FP16 plain/E4M3/E5M2, BF16 plain/E4M3/E5M2.
// Existing variants 0..13 retain their mapping. EP2 Plain16 appends FP16/BF16
// as 14/15 for the standalone single-host smoke test.
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
__aicore__ inline
#else
inline
#endif
constexpr uint32_t LlDispatchVariantForTiling(uint8_t dtype, uint8_t quantMode,
                                             uint8_t expandXOutDtype, uint8_t inputMode,
                                             uint8_t group)
{
    if (group == 2U) {
        if (quantMode != 0U || inputMode != kLlDispatchPlain16 ||
            (expandXOutDtype != dtype && expandXOutDtype != 255U)) return kLlDispatchInvalidVariant;
        return dtype == kLlDispatchFp16 ? 14U : (dtype == kLlDispatchBf16 ? 15U : kLlDispatchInvalidVariant);
    }
    if (group != 4U && group != 16U) return kLlDispatchInvalidVariant;
    uint32_t mode = kLlDispatchInvalidVariant;
    if (dtype == kLlDispatchE4M3 && quantMode == 4U &&
        expandXOutDtype == kLlDispatchE4M3 && inputMode == kLlDispatchPrequantized) {
        mode = 0U;
    } else if (dtype == kLlDispatchFp16 || dtype == kLlDispatchBf16) {
        const uint32_t base = dtype == kLlDispatchFp16 ? 1U : 4U;
        if (quantMode == 0U && inputMode == kLlDispatchPlain16 &&
            (expandXOutDtype == dtype || expandXOutDtype == 255U)) {
            mode = base;
        } else if (quantMode == 4U && inputMode == kLlDispatchQuantize16) {
            if (expandXOutDtype == kLlDispatchE4M3) mode = base + 1U;
            else if (expandXOutDtype == kLlDispatchE5M2) mode = base + 2U;
        }
    }
    return mode == kLlDispatchInvalidVariant ? mode : mode * 2U + (group == 16U ? 1U : 0U);
}
}  // namespace DispatchDedup
