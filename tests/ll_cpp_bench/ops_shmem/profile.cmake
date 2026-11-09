# Profile-only instrumentation of the pinned upstream algorithm. Fail closed if
# its source changes. Clean builds never apply these substitutions.
function(ops_profile_replace before after)
    string(FIND "${_text}" "${before}" _at)
    if(_at LESS 0)
        message(FATAL_ERROR "Missing ops profile anchor: ${before}")
    endif()
    string(LENGTH "${before}" _length)
    math(EXPR _end "${_at}+${_length}")
    string(SUBSTRING "${_text}" ${_end} -1 _rest)
    string(FIND "${_rest}" "${before}" _duplicate)
    if(NOT _duplicate LESS 0)
        message(FATAL_ERROR "Ambiguous ops profile anchor: ${before}")
    endif()
    string(REPLACE "${before}" "${after}" _text "${_text}")
    set(_text "${_text}" PARENT_SCOPE)
endfunction()

ops_profile_replace([=[    __aicore__ inline void Process();]=] [=[    __aicore__ inline void Process();
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* profile)
    {
        profile_.Bind(profile);
        hcomm_.SetProfile(profile);
    }
private:
    CombineProfile::View profile_;
public:]=])

ops_profile_replace([=[    if (rowStart == rowEnd) {]=] [=[    profile_.Set(CombineProfile::StageRows, rowEnd - rowStart);
    profile_.Mark(CombineProfile::StageSelected);
    if (rowStart == rowEnd) {]=])

ops_profile_replace([=[::BuildGatherReceivePlan(uint32_t mapCount)
{]=] [=[::BuildGatherReceivePlan(uint32_t mapCount)
{
    const auto profileBegin = profile_.DetailClock();
    profile_.Add(CombineProfile::RouteLoadedRows, mapCount);]=])
ops_profile_replace([=[    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t]=] [=[    profile_.Elapsed(CombineProfile::ReadyPlanTicks, profileBegin);
    profile_.Add(CombineProfile::ReadyCachedTasks, pendingCount);
    return pendingCount;
}

template <RoceCombineTypeClass>
__aicore__ inline uint32_t]=])

ops_profile_replace([=[::PollGatherToken(uint32_t mapIndex)
{]=] [=[::PollGatherToken(uint32_t mapIndex)
{
    const auto profileBegin = profile_.DetailClock();]=])
ops_profile_replace([=[    waitMasks.SetValue(mapIndex, waitMask);
    return waitMask == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateGatherToken]=] [=[    waitMasks.SetValue(mapIndex, waitMask);
    profile_.Elapsed(CombineProfile::ReadyPollCheckTicks, profileBegin);
    profile_.Add(CombineProfile::ReadyPollCalls);
    profile_.Add(CombineProfile::ReadyPollMisses, waitMask != 0U);
    if (!waitMask) profile_.MarkOnce(CombineProfile::ReadyFirstTask);
    return waitMask == 0U;
}

template <RoceCombineTypeClass>
__aicore__ inline void MoeDistributeCombineV2Roce<RoceCombineTypeFunc>::AccumulateGatherToken]=])
ops_profile_replace([=[    AccumulateReceiveInput<true>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM);
}]=] [=[    AccumulateReceiveInput<true>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM);
    profile_.Add(CombineProfile::PartialLocalContrib, issuedCount);
}]=])
ops_profile_replace([=[                AccumulateGatherToken(mapIndex);
                WriteGatherResult(gatherServerId_, GetGatherTokenId(mapIndex), resultCount++ % PAYLOAD_BUFFER_NUM);]=] [=[                const auto profileReduceBegin = profile_.DetailClock();
                AccumulateGatherToken(mapIndex);
                profile_.Elapsed(CombineProfile::PartialComputeIssueTicks, profileReduceBegin);
                const auto profileStoreBegin = profile_.DetailClock();
                WriteGatherResult(gatherServerId_, GetGatherTokenId(mapIndex), resultCount++ % PAYLOAD_BUFFER_NUM);
                profile_.Elapsed(CombineProfile::PartialPackTicks, profileStoreBegin);]=])
ops_profile_replace([=[    const uint32_t pendingCount = BuildGatherReceivePlan(mapCount);
    GatherReadyTokens(pendingCount);]=] [=[    const uint32_t pendingCount = BuildGatherReceivePlan(mapCount);
    profile_.Add(CombineProfile::PartialRows, pendingCount);
    profile_.MarkOnce(CombineProfile::PartialInitEnd);
    GatherReadyTokens(pendingCount);
    profile_.Mark(CombineProfile::ReadyTasksDone);]=])

ops_profile_replace([=[        if (!IsReceiveSlotReady(source)) {
            continue;
        }]=] [=[        const auto profileBegin = profile_.DetailClock();
        const bool profileReady = IsReceiveSlotReady(source);
        profile_.Elapsed(CombineProfile::ReadyPollCheckTicks, profileBegin);
        profile_.Add(CombineProfile::ReadyPollCalls);
        profile_.Add(CombineProfile::ReadyPollMisses, !profileReady);
        if (!profileReady) {
            continue;
        }
        profile_.MarkOnce(CombineProfile::ReadyFirstTask);]=])

ops_profile_replace([=[        ScatterTokensToRemote();
        return;]=] [=[        profile_.Set(CombineProfile::OpsRole, 3U);
        profile_.Mark(CombineProfile::DirectBegin);
        ScatterTokensToRemote();
        profile_.Mark(CombineProfile::DirectEnd);
        return;]=])
ops_profile_replace([=[    const GatherCoreAssignment assignment = GetGatherCoreAssignment();]=] [=[    profile_.Set(CombineProfile::OpsRole, 2U);
    const GatherCoreAssignment assignment = GetGatherCoreAssignment();]=])
ops_profile_replace([=[    gatherTokenStart_ = assignment.tokenStart;]=] [=[    profile_.Mark(CombineProfile::PartialBegin);
    gatherTokenStart_ = assignment.tokenStart;]=])
ops_profile_replace([=[        GatherServerTokens(server);
    }
}]=] [=[        GatherServerTokens(server);
    }
    profile_.Mark(CombineProfile::PartialEnd);
}]=])

ops_profile_replace([=[::PollFinalToken(uint32_t token)
{]=] [=[::PollFinalToken(uint32_t token)
{
    const auto profileBegin = profile_.DetailClock();]=])
ops_profile_replace([=[    waitMasks.SetValue(token, waitMask);
    return waitMask == 0U;]=] [=[    waitMasks.SetValue(token, waitMask);
    profile_.Probe(waitMask == 0U, finalTokenBegin_ + token, 0xffffffffU, 0U, profileBegin);
    return waitMask == 0U;]=])
ops_profile_replace([=[    if (issuedCount != 0U) {
        AccumulateReceiveInput<false>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM, pendingWeight);
    }
}]=] [=[    if (issuedCount != 0U) {
        AccumulateReceiveInput<false>((issuedCount - 1U) % PAYLOAD_BUFFER_NUM, pendingWeight);
    }
    profile_.Add(CombineProfile::Contributions, issuedCount);
}]=])
ops_profile_replace([=[            WriteFinalOutput(token, outputCount % PAYLOAD_BUFFER_NUM);]=] [=[            WriteFinalOutput(token, outputCount % PAYLOAD_BUFFER_NUM);
            profile_.Mark(CombineProfile::FinalLastStore);]=])
ops_profile_replace([=[            ReduceFinalToken(token);
            WriteFinalOutput(token, outputCount % PAYLOAD_BUFFER_NUM);]=] [=[            const auto profileReduceBegin = profile_.DetailClock();
            ReduceFinalToken(token);
            profile_.Elapsed(CombineProfile::FinalComputeIssueTicks, profileReduceBegin);
            const auto profileStoreBegin = profile_.DetailClock();
            WriteFinalOutput(token, outputCount % PAYLOAD_BUFFER_NUM);
            profile_.Elapsed(CombineProfile::FinalStoreIssueTicks, profileStoreBegin);]=])
ops_profile_replace([=[    const uint32_t tokenCount = tokenEnd - finalTokenBegin_;]=] [=[    const uint32_t tokenCount = tokenEnd - finalTokenBegin_;
    profile_.Set(CombineProfile::FinalTokens, tokenCount);]=])
ops_profile_replace([=[    const uint32_t pendingCount = BuildFinalReceivePlan(tokenCount);]=] [=[    const uint32_t pendingCount = BuildFinalReceivePlan(tokenCount);
    profile_.Mark(CombineProfile::FinalInitEnd);]=])

ops_profile_replace([=[        if (aivId_ < sendAivNum_) {
            ScatterTokenToLocal();
        } else {
            GatherTokenFromLocal();
        }
        SyncAll<true>();
        tpipe_->Reset();
        ReceiveAndCombine();]=] [=[        if (aivId_ < sendAivNum_) {
            profile_.Set(CombineProfile::OpsRole, 1U);
            profile_.Mark(CombineProfile::StageBegin);
            ScatterTokenToLocal();
            profile_.Mark(CombineProfile::StageEnd);
        } else {
            GatherTokenFromLocal();
        }
        profile_.Mark(CombineProfile::ReadyBegin);
        SyncAll<true>();
        profile_.Mark(CombineProfile::ReadyEnd);
        tpipe_->Reset();
        profile_.Mark(CombineProfile::FinalBegin);
        ReceiveAndCombine();
        profile_.Mark(CombineProfile::FinalEnd);]=])
