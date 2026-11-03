#pragma once
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "../../csrc/kernels/common/ep_server_order.h"

namespace llbench {
// Decode the actual Dispatch tables, not a reconstructed random route. The
// ordinal matches DIRECT/M4's stable group-order, row-order round robin.
inline std::vector<uint64_t> return_producers(const std::vector<int32_t>& types,
    const std::vector<int32_t>& table, uint32_t rank, uint32_t world, uint32_t bs,
    uint32_t k, uint32_t group, uint32_t hostRanks, uint32_t cores, uint32_t recv, bool remoteFirst)
{
    if (table.size() != types.size() * 3 || !recv || recv >= cores || !group || world % group)
        throw std::runtime_error("Invalid return producer table shape");
    std::vector<uint64_t> result(size_t(world) * bs * k);
    const uint32_t groups = world / group, own = rank / group;
    uint32_t directOrdinal = 0, partialOrdinal = 0;
    std::vector<uint32_t> order;
    for (uint32_t i = 0; i < groups; ++i) order.push_back(EpServerOrder::ServerDedupHalfOrder(groups, own, i));
    if (remoteFirst && group == hostRanks && groups > 1) {
        auto it = std::find(order.begin(), order.end(), own);
        order.erase(it); order.push_back(own);
    }
    for (uint32_t server : order) {
        for (size_t row = 0; row < types.size(); ++row) {
            const int32_t type = types[row];
            if (type != 1 && type != 2) continue;
            const uint32_t source = uint32_t(table[row * 3]), token = uint32_t(table[row * 3 + 1]);
            const uint32_t slot = uint32_t(table[row * 3 + 2]);
            if (source >= world || token >= bs || slot >= k) throw std::runtime_error("Invalid producer routing triple");
            if (source / group != server) continue;
            const uint32_t core = type == 1 ? directOrdinal++ % (cores - recv) : cores - recv + partialOrdinal++ % recv;
            auto& value = result[(size_t(source) * bs + token) * k + slot];
            if (value) throw std::runtime_error("Duplicate local return producer");
            value = uint64_t(rank + 1U) | (uint64_t(core) << 16U) | (uint64_t(type) << 24U) | (uint64_t(row) << 32U);
        }
    }
    return result;
}
}  // namespace llbench
