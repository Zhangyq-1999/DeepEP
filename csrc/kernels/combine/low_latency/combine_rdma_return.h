#pragma once
#include "../../dispatch/low_latency/device/ll_dispatch_rdma_batch.h"
#include "combine_return_schedule.h"
#include "combine_ready_layout.h"
#if DEEPEP_LL_COMBINE_READY >= 9 && DEEPEP_LL_COMBINE_READY <= 13
#include "combine_transfer_batch.h"
#endif
#if DEEPEP_LL_COMBINE_READY == 5
#include "combine_pair_staging.h"
#endif
#if defined(DEEPEP_LL_COMBINE_COALESCE) && DEEPEP_LL_COMBINE_COALESCE > 0
#include "combine_coalesce.h"
#endif
namespace CombineDedup {
#if defined(DEEPEP_LL_COMBINE_COALESCE) && DEEPEP_LL_COMBINE_COALESCE > 0
using ReturnBatch = CoalescedReturn;
#elif DEEPEP_LL_COMBINE_READY == 5
using ReturnBatch = PairStagingBatch;
#elif DEEPEP_LL_COMBINE_READY >= 9 && DEEPEP_LL_COMBINE_READY <= 13
using ReturnBatch = TransferBatch;
#else
using ReturnBatch = LlTransport::DispatchRdmaBatch;
#endif
// Caller has packed this output and acquired V->MTE3. A pending older frame
// uses a different output lane; single-lane callers must select defer=false.
__aicore__ inline void SendReturnPacket(ReturnBatch& batch,
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
