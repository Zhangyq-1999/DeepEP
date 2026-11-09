#include "kernel_operator.h"
#include "config.hpp"
#include "../../../csrc/kernels/launch.hpp"
#include "mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h"
#include <cstddef>

#ifdef DEEPEP_LL_COMBINE_PROFILE
#define OPS_PROFILE_PARAMS , GM_ADDR stageTimer, uint32_t timerMode
#define OPS_PROFILE_ARGS , stageTimer, timerMode
#define OPS_PROFILE_LAUNCH_ARGS , a.stageTimer, a.timerMode
#else
#define OPS_PROFILE_PARAMS
#define OPS_PROFILE_ARGS
#define OPS_PROFILE_LAUNCH_ARGS
#endif

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
static_assert(sizeof(opsbench::Context) == sizeof(Mc2Aclnn::Mc2MoeContext));
static_assert(offsetof(opsbench::Context, localWindows) == offsetof(Mc2Aclnn::Mc2MoeContext, epHcclBuffer_));
static_assert(offsetof(opsbench::Context, peers) == offsetof(Mc2Aclnn::Mc2MoeContext, hcommHandle_));
static_assert(offsetof(opsbench::Context, remoteWindows) == offsetof(Mc2Aclnn::Mc2MoeContext, roceHcclBuffer_));

template <class T>
__aicore__ inline void run_ops(GM_ADDR context, GM_ADDR expanded, GM_ADDR experts, GM_ADDR assist, GM_ADDR counts,
                               GM_ADDR weights, GM_ADDR expandedWeights, GM_ADDR output, GM_ADDR config OPS_PROFILE_PARAMS)
{
    const auto* c = reinterpret_cast<const __gm__ opsbench::Config*>(config);
    util_set_ffts_config(c->ffts);
#ifdef DEEPEP_LL_COMBINE_PROFILE
    CombineProfile::DeviceProfile profile;
    profile.Init(stageTimer, timerMode);
    profile.Mark(CombineProfile::MagicEnd);
#endif
    MoeDistributeCombineV2TilingData tiling{};
    auto& t = tiling.moeDistributeCombineV2Info;
    t.epWorldSize = c->world;
    t.epRankId = c->rank;
    t.moeExpertNum = c->experts;
    t.moeExpertPerRankNum = c->experts / c->world;
    t.bs = c->bs;
    t.globalBs = c->bs * c->world;
    t.h = c->hidden;
    t.k = c->topk;
    t.a = c->rows;
    t.aivNum = c->cores;
    t.totalUbSize = c->ubBytes;
    t.totalWinSizeEp = opsbench::windowBytes - opsbench::stateBytes;
    t.hasExpertScales = true;
    t.enablePreReduce = true;
    AscendC::TPipe pipe;
    MoeDistributeCombineV2RoceImpl::MoeDistributeCombineV2Roce<T, T, int32_t, 0, false> op;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    op.SetProfile(&profile);
#endif
    // HasAddRmsNorm=false: upstream writes XOut (argument 19), not yOut.
    op.Init(context, expanded, experts, assist, counts, nullptr, nullptr, weights, expandedWeights, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output, nullptr, &pipe, &tiling);
#ifdef DEEPEP_LL_COMBINE_PROFILE
    profile.Mark(CombineProfile::InitEnd);
#endif
    op.Process();
#ifdef DEEPEP_LL_COMBINE_PROFILE
    profile.Mark(CombineProfile::FlushDone);
    profile.Finish(c->rank, c->world, c->bs);
#endif
}
#endif

template <class T>
[[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ops_shmem_combine(GM_ADDR context, GM_ADDR expanded,
                                                                           GM_ADDR experts, GM_ADDR assist,
                                                                           GM_ADDR counts, GM_ADDR weights,
                                                                           GM_ADDR expandedWeights, GM_ADDR output,
                                                                           GM_ADDR config OPS_PROFILE_PARAMS)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    run_ops<T>(context, expanded, experts, assist, counts, weights, expandedWeights, output, config OPS_PROFILE_ARGS);
#endif
}
extern "C" DEEPEP_EXPORT void ops_shmem_combine_do(void* stream, const opsbench::Args* args)
{
    const auto& a = *args;
#define OPS_RUN(T)                                                         \
    ops_shmem_combine<T><<<a.cores, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>( \
        a.context, a.expanded, a.experts, a.assist, a.counts, a.weights, a.expandedWeights, a.output, a.config OPS_PROFILE_LAUNCH_ARGS)
    if (a.bf16) {
        OPS_RUN(bfloat16_t);
    } else {
        OPS_RUN(half);
    }
#undef OPS_RUN
}
#undef OPS_PROFILE_PARAMS
#undef OPS_PROFILE_ARGS
#undef OPS_PROFILE_LAUNCH_ARGS
