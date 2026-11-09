# Apply only to distinct experimental variants, after optional instrumentation.
# Pinned baseline remains exactly the four transport substitutions.
function(ops_experiment_replace before after)
    string(FIND "${_text}" "${before}" _at)
    if(_at LESS 0)
        message(FATAL_ERROR "Missing ops experiment anchor: ${before}")
    endif()
    string(LENGTH "${before}" _length)
    math(EXPR _end "${_at}+${_length}")
    string(SUBSTRING "${_text}" ${_end} -1 _rest)
    string(FIND "${_rest}" "${before}" _duplicate)
    if(NOT _duplicate LESS 0)
        message(FATAL_ERROR "Ambiguous ops experiment anchor: ${before}")
    endif()
    string(REPLACE "${before}" "${after}" _text "${_text}")
    set(_text "${_text}" PARENT_SCOPE)
endfunction()

ops_experiment_replace([=[    __aicore__ inline void Process();]=] [=[    __aicore__ inline void Process();
    __aicore__ inline void SetSmallExperiment(bool enabled)
    {
        smallExperiment_ = enabled;
        hcomm_.EnableDeferred(enabled);
    }
private:
    bool smallExperiment_ = false;
    __aicore__ inline void RetireTransport()
    {
        if (!hcomm_.HasPending()) return;
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile_.Mark(CombineProfile::ReadyRetireBegin);
#endif
        tpipe_->Reset();
        TBuf<> completionScratch;
        tpipe_->InitBuffer(completionScratch, 128U);
        hcomm_.Retire(completionScratch.Get<uint8_t>());
        tpipe_->Reset();
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile_.Mark(CombineProfile::ReadyRetireEnd);
#endif
    }
public:]=])

set(_barrier "        SyncAll<true>();")
if(DEEPEP_LL_COMBINE_PROFILE)
    set(_barrier "        profile_.Mark(CombineProfile::ReadyBegin);\n${_barrier}\n        profile_.Mark(CombineProfile::ReadyEnd);")
endif()
if(DEEPEP_LL_OPS_POLICY STREQUAL "1")
    ops_experiment_replace("${_barrier}" "        RetireTransport();\n${_barrier}")
elseif(DEEPEP_LL_OPS_POLICY GREATER_EQUAL 2)
    # Per-packet generation/flag checks already guard every final input, including
    # local-server results. Producers drain their own DMA before TPipe reuse.
    ops_experiment_replace("${_barrier}" "        if (!smallExperiment_) {\n${_barrier}\n        }")
    set(_final "        ReceiveAndCombine();")
    if(DEEPEP_LL_COMBINE_PROFILE)
        string(APPEND _final "\n        profile_.Mark(CombineProfile::FinalEnd);")
    endif()
    ops_experiment_replace("${_final}" "${_final}\n        if (smallExperiment_) {\n            RetireTransport();\n${_barrier}\n        }")
endif()
