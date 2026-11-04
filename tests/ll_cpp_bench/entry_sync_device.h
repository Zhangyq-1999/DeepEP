/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See ../../third-party/cann.LICENSE for the full text of the License.
 *
 * Adapted for the benchmark from vallenChen/shmem fix_1825_sync,
 * 71a2154261faa188e025e1654fd67b0a7b986a2b, aclshmemi_roce_team_sync<T, true>.
 * Only the world team and a single AIV are used here. Keep the installed SDK
 * and the measured Combine kernel unchanged; reuse the SDK signal primitives.
 */
#pragma once

ACLSHMEM_DEVICE void ll_bench_sync_world(__ubuf__ uint8_t* scratch)
{
    auto state = aclshmemi_get_state();
    auto team = state->team_pools[ACLSHMEM_TEAM_WORLD];
    auto pool = aclshmemi_get_team_sync_pool(team->team_idx);
    auto counter = aclshmemi_get_team_sync_counter(team->team_idx);
    const int32_t generation = aclshmemi_load((__gm__ int32_t*)counter) + 1;
    aclshmemi_store((__gm__ int32_t*)counter, generation);
    dcci_cacheline((__gm__ uint8_t*)counter);

    const int local = (team->mype - team->start) / team->stride;
    for (int shift = 1, round = 0; shift < team->size; shift *= 2, ++round) {
        const int peer = team->start + ((local + shift) % team->size) * team->stride;
        auto dst = (__gm__ int32_t*)(pool + round);
        const uint8_t transport = state->topo_list[peer];
        if (transport & ACLSHMEM_TRANSPORT_MTE) {
            // MTE wins when both bits are present. Adjacent cards on one 1825
            // need this path; pure RoCE all-rank synchronization is not valid.
            aclshmemi_signal_set((__gm__ int32_t*)aclshmem_ptr(dst, peer), generation);
        } else if (transport & ACLSHMEM_TRANSPORT_ROCE) {
            // SDK write + quiet retires the send before counter/scratch reuse.
            aclshmemi_highlevel_signal_set(dst, (__gm__ int32_t*)counter, peer, scratch, 0);
        } else {
            aclshmemi_kernel_abort("LL entry sync: no MTE or RoCE path to pe=%d.\n", peer);
            return;
        }
        // The SDK invalidates stale cache lines and accepts generation + 1
        // when a faster peer has already entered the next invocation.
        aclshmemi_signal_wait_until_eq_for_barrier(dst, generation);
    }
}
