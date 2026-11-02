// Host-side selection for separately compiled LL dispatch specializations.
#include <cstdint>
#include <stdexcept>
#include "../launch.hpp"
#include "low_latency/launch.hpp"
#include "low_latency/common/ll_core_config.h"
#include "low_latency/common/ll_group_policy.h"
#include "low_latency/common/ll_reference_profile_layout.h"

#if CATLASS_ARCH == 3510
extern "C" void dispatch_server_dedup_launch_fp16_g2(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_g2(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_prequant_e4m3_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_prequant_e4m3_g16(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_g16(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_e4m3_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_e4m3_g16(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_e5m2_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_fp16_e5m2_g16(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_g16(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_e4m3_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_e4m3_g16(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_e5m2_g4(void*, const deepep::EpServerDedupKernelArgs*);
extern "C" void dispatch_server_dedup_launch_bf16_e5m2_g16(void*, const deepep::EpServerDedupKernelArgs*);

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_stage_timer_enabled()
{
    return 0U;  // Legacy stage/expert device instrumentation has been removed.
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_reference_profile_enabled()
{
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    return 1U;
#else
    return 0U;
#endif
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_reference_profile_schema()
{
    return LlReferenceProfile::kSchema;
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_ranks_per_group(uint32_t world)
{
    return DispatchDedup::LlGroupForWorld(world);
}

extern "C" DEEPEP_EXPORT uint32_t deepep_ll_core_protocol_version()
{
    return DispatchDedup::kLlCoreProtocolVersion;
}
extern "C" DEEPEP_EXPORT uint32_t deepep_ll_rdma_enabled()
{
#ifdef DEEPEP_LL_RDMA
    return 1U;
#else
    return 0U;
#endif
}

extern "C" DEEPEP_EXPORT void dispatch_server_dedup_kernel_do_v2(
    void* stream, const deepep::EpServerDedupKernelArgs* args, uint32_t variant)
{
    if (!args) throw std::invalid_argument("LL dispatch launch arguments are null");
    switch (variant) {
    case 0U: dispatch_server_dedup_launch_prequant_e4m3_g4(stream, args); return;
    case 1U: dispatch_server_dedup_launch_prequant_e4m3_g16(stream, args); return;
    case 2U: dispatch_server_dedup_launch_fp16_g4(stream, args); return;
    case 3U: dispatch_server_dedup_launch_fp16_g16(stream, args); return;
    case 4U: dispatch_server_dedup_launch_fp16_e4m3_g4(stream, args); return;
    case 5U: dispatch_server_dedup_launch_fp16_e4m3_g16(stream, args); return;
    case 6U: dispatch_server_dedup_launch_fp16_e5m2_g4(stream, args); return;
    case 7U: dispatch_server_dedup_launch_fp16_e5m2_g16(stream, args); return;
    case 8U: dispatch_server_dedup_launch_bf16_g4(stream, args); return;
    case 9U: dispatch_server_dedup_launch_bf16_g16(stream, args); return;
    case 10U: dispatch_server_dedup_launch_bf16_e4m3_g4(stream, args); return;
    case 11U: dispatch_server_dedup_launch_bf16_e4m3_g16(stream, args); return;
    case 12U: dispatch_server_dedup_launch_bf16_e5m2_g4(stream, args); return;
    case 13U: dispatch_server_dedup_launch_bf16_e5m2_g16(stream, args); return;
    case 14U: dispatch_server_dedup_launch_fp16_g2(stream, args); return;
    case 15U: dispatch_server_dedup_launch_bf16_g2(stream, args); return;
    default: throw std::invalid_argument("Invalid LL dispatch specialization");
    }
}
#endif
