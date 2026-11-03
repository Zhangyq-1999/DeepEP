#include "kernel_operator.h"
#include "shmem.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "../../csrc/kernels/launch.hpp"
#include "entry_sync_device.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ll_bench_entry_sync_kernel(
    uint64_t ffts, uint64_t waitTicks)
{
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, 512U);
    ll_bench_sync_world(reinterpret_cast<__ubuf__ uint8_t*>(scratch.Get<uint8_t>().GetPhyAddr()), waitTicks);
    AscendC::PipeBarrier<PIPE_ALL>();
}

extern "C" DEEPEP_EXPORT void ll_bench_entry_sync_do(void* stream, uint64_t ffts, uint64_t waitTicks)
{
    ll_bench_entry_sync_kernel<<<1, DEEPEP_KERNEL_LAUNCH_UB(1024), stream>>>(ffts, waitTicks);
}
