#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace llbench {
// Match CPython Random(integer), random(), choices() and shuffle(). A plain
// std::mt19937(seed) uses a different seed expansion and cannot reproduce routes.
class RouteRandom {
    struct Seed {
        using result_type = uint32_t;
        uint64_t value;
        template<class It> void generate(It first, It /*last*/)
        {
            std::array<uint32_t, 624> state;
            state[0] = 19650218U;
            for (uint32_t i = 1; i < state.size(); ++i)
                state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
            const uint32_t key[] = {uint32_t(value), uint32_t(value >> 32)};
            const uint32_t words = key[1] ? 2U : 1U;
            uint32_t i = 1, j = 0;
            for (uint32_t n = 624; n; --n) {
                state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525U)) + key[j] + j;
                if (++i == 624) { state[0] = state[623]; i = 1; }
                if (++j == words) j = 0;
            }
            for (uint32_t n = 623; n; --n) {
                state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941U)) - i;
                if (++i == 624) { state[0] = state[623]; i = 1; }
            }
            state[0] = 0x80000000U;
            std::copy(state.begin(), state.end(), first);
        }
    };
    std::mt19937 engine_;
public:
    explicit RouteRandom(uint64_t seed) { Seed sequence{seed}; engine_.seed(sequence); }
    double random()
    {
        const uint64_t high = engine_() >> 5, low = engine_() >> 6;
        return double((high << 26) + low) / 9007199254740992.0;
    }
    uint32_t below(uint32_t n)
    {
        uint32_t bits = 0;
        for (auto v = n; v; v >>= 1) ++bits;
        uint32_t value;
        do { value = uint32_t(engine_()) >> (32 - bits); } while (value >= n);
        return value;
    }
};

// Global source-rank/token order matches make_random_expert_indices() in
// tests/elastic/test_ep_low_latency.py (normal, hot_multiplier_x2=2).
// Host fixture preparation only; no work is added to the measured kernel.
inline std::vector<int32_t> normal_routes(uint32_t world, uint32_t bs, uint32_t topk,
                                          uint32_t experts, uint32_t seed)
{
    if (!world || !bs || !topk || topk > world || experts < world || experts % world)
        throw std::invalid_argument("normal routing requires BS>0, E divisible by EP, and 1<=TopK<=EP");
    RouteRandom rng(20261014ULL + seed + uint64_t(bs) * 1000U + 2U);
    std::vector<uint32_t> remaining(world, bs * topk), occurrences(world, 0);
    std::vector<int32_t> result;
    result.reserve(size_t(world) * bs * topk);
    std::vector<uint32_t> chosen;
    chosen.reserve(topk);
    std::vector<bool> used(world);
    for (uint32_t token = 0; token < world * bs; ++token) {
        chosen.clear();
        std::fill(used.begin(), used.end(), false);
        for (uint32_t r = 0; r < world; ++r)
            if (remaining[r] == world * bs - token) { chosen.push_back(r); used[r] = true; }
        while (chosen.size() < topk) {
            uint32_t total = 0;
            for (uint32_t r = 0; r < world; ++r) if (!used[r]) total += remaining[r];
            const double pick = rng.random() * total;
            uint32_t cumulative = 0;
            for (uint32_t r = 0; r < world; ++r) {
                if (used[r]) continue;
                cumulative += remaining[r];
                if (pick < cumulative) { chosen.push_back(r); used[r] = true; break; }
            }
        }
        for (uint32_t n = uint32_t(chosen.size()); n > 1; --n)
            std::swap(chosen[n - 1], chosen[rng.below(n)]);
        for (auto r : chosen) {
            result.push_back(int32_t(r * (experts / world) + occurrences[r] % (experts / world)));
            --remaining[r];
            ++occurrences[r];
        }
    }
    return result;
}
} // namespace llbench

// MT19937 init_by_array seed expansion above is adapted from the reference
// algorithm used by CPython Modules/_randommodule.c.
// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
