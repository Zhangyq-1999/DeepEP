// Benchmark-only stream gate. Combine itself is unchanged by launch alignment.
#include "kernel_operator.h"
#include "shmem.h"
#include "../../csrc/kernels/launch.hpp"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_bench_entry_gate(GM_ADDR address)
{
    AscendC::GlobalTensor<uint32_t> flag;
    flag.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    flag.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    const uint64_t begin = AscendC::GetSystemCycle();
    while (true) {
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(flag);
        if (flag.GetValue(0) == 1U) return;
        if (AscendC::GetSystemCycle() - begin >= 120000000000ULL) {
            AscendC::Trap();
            return;
        }
    }
}
extern "C" DEEPEP_EXPORT void ll_bench_entry_gate_do(void* stream, uint8_t* flag)
{
    ll_bench_entry_gate<<<1, DEEPEP_KERNEL_LAUNCH_UB(0), stream>>>(flag);
}

// Separate pre-Event kernel. The pinned SDK implements a dissemination sync
// over the initialized RoCE peer QPs. Combine has
// already retired its queues at the preceding stream synchronization.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_bench_roce_entry_sync()
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 128U);
    auto scratch = buffer.Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_ALL>();
    aclshmemx_roce_sync_all(reinterpret_cast<__ubuf__ uint8_t*>(scratch.GetPhyAddr()), 0);
    AscendC::PipeBarrier<PIPE_ALL>();
}
extern "C" DEEPEP_EXPORT void ll_bench_roce_entry_sync_do(void* stream)
{
    ll_bench_roce_entry_sync<<<1, DEEPEP_KERNEL_LAUNCH_UB(128), stream>>>();
}
