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
    __aicore__ inline void Init(GM_ADDR address, uint32_t level)
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
    __aicore__ inline void Set(Word word, uint64_t value)
    {
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (record_) values_[word] = value;
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
