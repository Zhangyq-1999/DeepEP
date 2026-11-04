#pragma once
#include <cstdint>
#include "../../dispatch/low_latency/common/ll_core_config.h"
#include "../../dispatch/low_latency/common/ep_memory_server_dedup_protocol.h"
#include "ep_memory_server_dedup_buffer_layout.h"

#ifndef DEEPEP_LL_COMBINE_NOTIFY
#define DEEPEP_LL_COMBINE_NOTIFY 0
#endif

namespace CombineDedup {
// Each consumer owns a separate row; no reader clears another reader's flags.
constexpr uint32_t kCoreReadyPeerWords = DispatchDedup::kLlMaxCoreNum;
constexpr uint32_t kCoreReadyRowBytes = DispatchDedup::kLlRanksPerGroup * kCoreReadyPeerWords * 4U;
static_assert(COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CORE_READY_OFFSET +
              DispatchDedup::kLlMaxRecvCoreNum * kCoreReadyRowBytes < 1024U * 1024U,
              "core-ready table must remain below the payload boundary");
}  // namespace CombineDedup
