#pragma once
#ifndef DEEPEP_LL_OPS_POLICY
#define DEEPEP_LL_OPS_POLICY 0
#endif
static_assert(DEEPEP_LL_OPS_POLICY >= 0 && DEEPEP_LL_OPS_POLICY <= 4, "invalid ops policy");
#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define OPS_POLICY_CONSTEXPR __aicore__ inline constexpr
#else
#define OPS_POLICY_CONSTEXPR constexpr
#endif
namespace opsbench {
constexpr unsigned policy = DEEPEP_LL_OPS_POLICY;
constexpr const char* implementation = policy == 1 ? "ops-shmem-retire" :
    policy == 2 ? "ops-shmem-overlap" : policy == 3 ? "ops-shmem-cq" :
    policy == 4 ? "ops-shmem-batch" : "ops-shmem-base";
constexpr const char* transportDescription = policy == 0 ? "SHMEM per-core QP Put; upstream batch quiet" :
    policy == 1 ? "SHMEM immediate Put; tiny-BS quiet once per peer before barrier" :
    policy == 2 ? "SHMEM immediate Put; tiny-BS final combine before quiet and exit barrier" :
    policy == 3 ? "SHMEM immediate Put; tiny-BS ld_dev CQ retirement after final" :
                  "SHMEM QP/MR; tiny-BS UB WQE batch + one DB + ld_dev CQ retirement after final";
// Pre-reduce emits at most BS immutable source slots per peer. Bound experiments
// to 24 posts/peer and 64 PEs; larger shapes retain the upstream schedule.
OPS_POLICY_CONSTEXPR bool smallShape(unsigned bs, unsigned world) { return bs <= 24U && world <= 64U; }
}  // namespace opsbench
#undef OPS_POLICY_CONSTEXPR
