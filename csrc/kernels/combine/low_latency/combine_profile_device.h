#pragma once
#include "kernel_operator.h"
#include "combine_profile_layout.h"
#ifdef DEEPEP_LL_COMBINE_PROFILE
#include "shmem.h"
#endif

namespace CombineProfile {
// No new synchronization in the measured path. Accumulate in local scalar state
// and export once, after the original kernel's final drain/flush.
class DeviceProfile {
public:
    __aicore__ inline void Init(__gm__ uint8_t* address, uint32_t level)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        record_ =
            address && level ? reinterpret_cast<__gm__ uint64_t*>(address) + AscendC::GetBlockIdx() * kWords : nullptr;
        level_ = record_ ? level : 0U;
        if (record_) {
            const uint64_t entry = AscendC::GetSystemCycle();
            for (uint32_t i = 0; i < kWords; ++i) values_[i] = 0U;
            values_[Entry] = entry;
            values_[Valid] = 1U;
            Mark(ProfileInitEnd);
        }
#endif
    }
    __aicore__ inline uint64_t DetailClock() const
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (level_ >= 2U) return AscendC::GetSystemCycle();
#endif
        return 0U;
    }
    __aicore__ inline void Mark(Word word)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (record_) {
            values_[word] = AscendC::GetSystemCycle();
            values_[Valid] |= uint64_t(1) << word;
        }
#endif
    }
    __aicore__ inline void MarkOnce(Word word)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (record_ && !(values_[Valid] & (uint64_t(1) << word))) Mark(word);
#endif
    }
    __aicore__ inline void SampleMax(Word total, Word maximum, Word id0, Word id1, uint64_t begin, uint32_t a,
                                     uint32_t b)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (level_ < 2U) return;
        const uint64_t ticks = AscendC::GetSystemCycle() - begin;
        values_[total] += ticks;
        if (ticks > values_[maximum]) {
            values_[maximum] = ticks;
            values_[id0] = a;
            values_[id1] = b;
        }
#endif
    }
    __aicore__ inline void Probe(bool ready, uint32_t token, uint32_t slot, uint32_t blocks, uint64_t begin)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (!record_) return;
        if (level_ >= 2U) {
            const uint64_t ticks = AscendC::GetSystemCycle() - begin;
            values_[ProbeTicks] += ticks;
            ++values_[ProbeCalls];
            values_[ProbeFlagBlocks] += blocks;
            if (!ready) {
                ++values_[ProbeMisses];
                values_[ProbeMissTicks] += ticks;
                values_[ProbeBlockedToken] = token;
                values_[ProbeBlockedSlot] = slot;
            }
            if (ticks > values_[ProbeMaxTicks]) {
                values_[ProbeMaxTicks] = ticks;
                values_[ProbeMaxToken] = token;
                values_[ProbeMaxSlot] = slot;
            }
        }
        if (ready) {
            MarkOnce(FinalFirstReady);
            Mark(FinalLastReady);
        }
#endif
    }
    __aicore__ inline void Set(Word word, uint64_t value)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (record_) values_[word] = value;
#endif
    }
    __aicore__ inline void Missing(uint32_t token, uint32_t mask, uint32_t first, uint32_t count, uint32_t blocks)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (level_ < 2U) return;
        const uint64_t tick = AscendC::GetSystemCycle() - values_[Entry];
        if (!mask) {
            if (values_[TailObservations] && values_[TailToken] == token && !values_[TailReady])
                values_[TailReady] = tick;
            return;
        }
        const uint32_t slot = first / blocks;
        if (!values_[TailObservations] || values_[TailToken] != token || values_[TailSlot] != slot) {
            values_[TailFirstSeen] = tick;
            values_[TailObservations] = 0U;
        }
        values_[TailToken] = token;
        values_[TailMissingMask] = mask;
        values_[TailSlot] = slot;
        values_[TailBlock] = first % blocks;
        values_[TailMissingBlocks] = count;
        values_[TailLastSeen] = tick;
        values_[TailReady] = 0U;
        ++values_[TailObservations];
#endif
    }
    __aicore__ inline void Add(Word word, uint64_t value = 1U)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (level_ >= 2U) values_[word] += value;
#endif
    }
    __aicore__ inline void Elapsed(Word word, uint64_t begin)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (level_ >= 2U) values_[word] += AscendC::GetSystemCycle() - begin;
#endif
    }
    __aicore__ inline void Finish(uint32_t rank, uint32_t world, uint32_t tokens)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (!record_) return;
        Set(Rank, rank);
        Set(Core, AscendC::GetBlockIdx());
        Set(Tokens, tokens);
        Set(World, world);
        Set(Schema, kSchema);
        Set(Level, level_);
        Set(Complete, 1U);
        for (uint32_t i = 0; i < kWords; ++i) record_[i] = values_[i];
        // Match the existing Dispatch diagnostic export; no adjacent core's line.
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(record_), kWords * sizeof(uint64_t) - 1U);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
#endif
    }

private:
#ifdef DEEPEP_LL_COMBINE_PROFILE
    __gm__ uint64_t* record_ = nullptr;
    uint32_t level_ = 0U;
    uint64_t values_[kWords];
#endif
};

// Optional shared view for helpers also exercised independently by CPU models.
class View {
public:
    __aicore__ inline void Missing(uint32_t token, uint32_t mask, uint32_t first, uint32_t count, uint32_t blocks)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->Missing(token, mask, first, count, blocks);
#endif
    }
    __aicore__ inline void Bind(DeviceProfile* profile)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile_ = profile;
#endif
    }
    __aicore__ inline uint64_t DetailClock() const
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) return profile_->DetailClock();
#endif
        return 0U;
    }
    __aicore__ inline void Mark(Word word)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->Mark(word);
#endif
    }
    __aicore__ inline void MarkOnce(Word word)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->MarkOnce(word);
#endif
    }
    __aicore__ inline void SampleMax(Word total, Word maximum, Word id0, Word id1, uint64_t begin, uint32_t a,
                                     uint32_t b)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->SampleMax(total, maximum, id0, id1, begin, a, b);
#endif
    }
    __aicore__ inline void Probe(bool ready, uint32_t token, uint32_t slot, uint32_t blocks, uint64_t begin)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->Probe(ready, token, slot, blocks, begin);
#endif
    }
    __aicore__ inline void Set(Word word, uint64_t value)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->Set(word, value);
#endif
    }
    __aicore__ inline void Add(Word word, uint64_t value = 1U)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->Add(word, value);
#endif
    }
    __aicore__ inline void Elapsed(Word word, uint64_t begin)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (profile_) profile_->Elapsed(word, begin);
#endif
    }

private:
#ifdef DEEPEP_LL_COMBINE_PROFILE
    DeviceProfile* profile_ = nullptr;
#endif
};
}  // namespace CombineProfile
