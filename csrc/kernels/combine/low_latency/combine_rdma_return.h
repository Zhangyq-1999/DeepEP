#pragma once
#include "../../dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
#include "combine_return_schedule.h"
namespace CombineDedup {
// Caller has packed this output and acquired V->MTE3. A pending older frame
// uses a different output lane; single-lane callers must select defer=false.
__aicore__ inline void SendReturnPacket(LlTransport::DispatchRdmaBatch& batch,
                                        const __gm__ DispatchDedup::CommArgs* args,
                                        uint32_t peer, uint64_t offset,
                                        AscendC::LocalTensor<uint8_t> output,
                                        uint32_t bytes, bool defer)
{
    batch.SubmitPending();
    if (LlTransport::RemoteWrite(args, peer)) batch.Enqueue(peer, offset, output, defer);
    else LlTransport::Packet(args, peer, offset, output, bytes, 7U, false);
}
}
