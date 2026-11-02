// Compile exactly one dispatch specialization per translation unit.
#pragma once
#include <cstdint>
#include "kernel_operator.h"
#include "shmem.h"
#include "../../launch.hpp"
#include "dispatch_server_dedup_kernel.h"
#include "common/dispatch_dedup_types.h"
#include "common/ep_memory_server_dedup_tiling.h"
#include "common/ll_dispatch_variant.h"
#include "common/ll_reference_profile_device.h"
#include "launch.hpp"

#define DEEPEP_LL_JOIN_IMPL(a, b) a##b
#define DEEPEP_LL_JOIN(a, b) DEEPEP_LL_JOIN_IMPL(a, b)
#define DEEPEP_LL_VARIANT_KERNEL DEEPEP_LL_JOIN(dispatch_server_dedup_kernel_, DEEPEP_LL_VARIANT_NAME)
#define DEEPEP_LL_VARIANT_LAUNCH DEEPEP_LL_JOIN(dispatch_server_dedup_launch_, DEEPEP_LL_VARIANT_NAME)

#if CATLASS_ARCH == 3510
namespace {
#if DEEPEP_LL_VARIANT_MODE == 0
using LlVariantInput = uint8_t;
constexpr bool kWireFp8 = true, kE4M3 = true, kPrequantized = true;
#elif DEEPEP_LL_VARIANT_MODE == 1
using LlVariantInput = half;
constexpr bool kWireFp8 = false, kE4M3 = false, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 2
using LlVariantInput = half;
constexpr bool kWireFp8 = true, kE4M3 = true, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 3
using LlVariantInput = half;
constexpr bool kWireFp8 = true, kE4M3 = false, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 4
using LlVariantInput = bfloat16_t;
constexpr bool kWireFp8 = false, kE4M3 = false, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 5
using LlVariantInput = bfloat16_t;
constexpr bool kWireFp8 = true, kE4M3 = true, kPrequantized = false;
#elif DEEPEP_LL_VARIANT_MODE == 6
using LlVariantInput = bfloat16_t;
constexpr bool kWireFp8 = true, kE4M3 = false, kPrequantized = false;
#else
#error Invalid LL dispatch variant
#endif
static_assert(DEEPEP_LL_VARIANT_GROUP == 2 || DEEPEP_LL_VARIANT_GROUP == 4 || DEEPEP_LL_VARIANT_GROUP == 16, "Invalid LL group");
static_assert(DEEPEP_LL_VARIANT_GROUP != 2 || DEEPEP_LL_VARIANT_MODE == 1 || DEEPEP_LL_VARIANT_MODE == 4, "EP2 supports Plain16");
constexpr uint32_t kDispatchVariant = DEEPEP_LL_VARIANT_GROUP == 2
    ? (DEEPEP_LL_VARIANT_MODE == 1 ? 14U : 15U)
    : DEEPEP_LL_VARIANT_MODE * 2U + (DEEPEP_LL_VARIANT_GROUP == 16 ? 1U : 0U);
}

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void DEEPEP_LL_VARIANT_KERNEL(
    GM_ADDR commArgs, GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tokenTypeOut, GM_ADDR destinationIndexOut,
    GM_ADDR relayReadIndexOut, GM_ADDR sourceMaskOut, GM_ADDR expertScalesOut, GM_ADDR tilingData,
    int64_t reservedMagic, GM_ADDR inputScales, GM_ADDR stageTimer, uint32_t timerMode, GM_ADDR expertTimer)
{
    (void)timerMode; (void)expertTimer;  // Reserved launch ABI slots; legacy tracing removed.
    const uint64_t referenceEntry = LlReferenceProfile::DeviceProfile::EntryClock(stageTimer);
    util_set_ffts_config(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->fftsVal);
    LlReferenceProfile::DeviceProfile profile;
    profile.Init(stageTimer);
    profile.Value<0>(referenceEntry);  // w0: kernel_entry
    profile.Mark<1>();  // w1: config_ready

    // Host owns validation; each core reads its configuration once before work.
    const auto* tiling = reinterpret_cast<const __gm__ DispatchDedup::EpServerDedupTilingData*>(tilingData);
    if (!DispatchDedup::LlTilingKeyValid(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rankSize, tiling->tilingKey)) { AscendC::Trap(); return; }
    const uint32_t bs = tiling->bs;
    const uint32_t h = tiling->h;
    const bool enableDedup = DispatchDedup::EpServerDedupEnabled(tiling->numMaxTokensPerRank, bs,
        uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rankSize));
    const uint32_t topK = tiling->topK;
    const uint32_t moeExpertNum = tiling->moeExpertNum;

    const uint32_t expertTokenNumsType = tiling->expertTokenNumsType;
    const uint8_t quantMode = tiling->quantMode;
    const uint8_t dtype = tiling->dtype;
    const uint8_t expandXOutDtype = tiling->expandXOutDtype;
    const auto inputMode = static_cast<DispatchDedup::EpServerDedupInputMode>(tiling->inputMode);
    AscendC::TPipe pipe;

    // Host chooses the specialization without reading device memory. Verify the
    // uploaded tiling still agrees before executing the unchanged algorithm.
    if (DispatchDedup::LlDispatchVariantForTiling(dtype, quantMode, expandXOutDtype,
            static_cast<uint8_t>(inputMode), tiling->tilingKey) != kDispatchVariant) {
        AscendC::Trap(); return;
    }
    {
        // Preserve the original branch-local lifetime of SDK buffer members.
        DispatchDedup::DispatchMemoryServerDedupKernel<LlVariantInput, kWireFp8, kE4M3,
            kPrequantized, DEEPEP_LL_VARIANT_GROUP> kernel;
        kernel.Init(commArgs, x, expertIds, expertScales, expandXOut, dynamicScalesOut, expertTokenNumsOut, sendCountsOut,
                    tokenTypeOut, destinationIndexOut, relayReadIndexOut, sourceMaskOut, expertScalesOut, bs, h, topK,
                    moeExpertNum, expertTokenNumsType, &pipe, inputScales, enableDedup, &profile);

        kernel.Process();
    }

    profile.Finish(uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rank),
                   uint32_t(reinterpret_cast<const __gm__ DispatchDedup::CommArgs*>(commArgs)->rankSize), bs);
}

extern "C" DEEPEP_EXPORT void DEEPEP_LL_VARIANT_LAUNCH(void* stream, const deepep::EpServerDedupKernelArgs* args)
{
    const auto& a = *args;
    DEEPEP_LL_VARIANT_KERNEL<<<a.coreNum, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(
        a.commArgs, a.x, a.expertIds, a.expertScales, a.expandXOut, a.dynamicScalesOut, a.expertTokenNumsOut,
        a.sendCountsOut, a.tokenTypeOut, a.destinationIndexOut, a.relayReadIndexOut, a.sourceMaskOut, a.expertScalesOut,
        a.tilingData, a.reservedMagic, a.inputScales, a.stageTimer, a.timerMode, a.expertTimer);
}

#endif
