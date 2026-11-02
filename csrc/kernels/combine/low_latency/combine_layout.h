// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#pragma once
#include "ep_memory_server_dedup_combine_config.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "../../dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
namespace CombineDedup {
struct Layout {
    uint64_t capacityRows = 0, requiredWindowBytes = 0;
};
inline bool BuildLayout(int64_t p, int64_t b, int64_t h, int64_t k, int64_t e, Layout& out, uint32_t coreNum = DispatchDedup::kLlDefaultPlanCoreNum)
{
    out = {};
    if (!DispatchDedup::LlCoreCountValid(coreNum) || p < 2 || (p < 16 && p != 2 && p != 4 && p != 8) || p > DispatchDedup::DISPATCH_DEDUP_MAX_RANK_SIZE || p % DispatchDedup::LlGroupForWorld(uint32_t(p)) || b <= 0 || b > 65536 || h <= 0 ||
        h > kEpServerDedupCombineRecvUbBytes / 12 || k < 1 || k > 32 || e < p || e > 1024 || e % p || k > e)
        return false;
    const uint64_t capacity = p * b * (k < e / p ? k : e / p);
    const uint32_t peak = capacity < kEpServerDedupCombineMaxTokenNum ? capacity : kEpServerDedupCombineMaxTokenNum;
    const uint32_t recvCores = EpCombineRecvCoreCount(coreNum);
    if (!recvCores || !EpCombinePayloadLaneCount(peak, h) || !EpCombineM45Layout(peak, b, h, k, recvCores).inputLanes) return false;
    const uint64_t packet = (h * 2 + 479) / 480 * 512;
    const uint64_t bytes =
        COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + b * k * packet + e * b * packet;
    if (bytes > DispatchDedup::kEpServerDedupWindowUsableBytes ||
        DispatchDedup::DispatchServerDedupCountLayout(e, coreNum).stateBytes >
            COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET)
        return false;
    out.capacityRows = capacity;
    out.requiredWindowBytes = bytes;
    return true;
}
}  // namespace CombineDedup
