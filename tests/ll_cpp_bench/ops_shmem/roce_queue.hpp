#pragma once
// Experimental HNS-1825 data plane over SHMEM-owned, per-core QPs/MRs.
// Queue ownership, indices and registration remain SHMEM's. No HCCL calls.
#include "kernel_operator.h"
#include "shmem.h"
#include "device/gm2gm/engine/shmem_device_rdma.h"
#include "../../../csrc/kernels/combine/low_latency/combine_profile_device.h"
#include <cstddef>

namespace opsbench {
struct QueueBatch {
    __gm__ aclshmemi_rdma_sq_ctx* sq = nullptr;
    __gm__ aclshmemi_rdma_cq_ctx* cq = nullptr;
    AscendC::LocalTensor<uint32_t> storage;
    uint32_t head = 0, count = 0, capacity = 0;
    uint64_t localBase = 0, remoteBase = 0, keys = 0, dataKey = 0;
};

class RoceQueue {
    CombineProfile::View profile_;
    uint32_t maxBatch_ = 0;
    __aicore__ inline void Bind(uint32_t peer, __gm__ aclshmemi_rdma_sq_ctx*& sq,
                               __gm__ aclshmemi_rdma_cq_ctx*& cq)
    {
        static_assert(ACLSHMEMI_K_RDMA_BACKEND == aclshmemi_rdma_backend_t::HNS_1825,
                      "ops CQ/batch experiments require the pinned HNS-1825 backend");
        auto* info = aclshmemi_qp_info_fetch();
        if (AscendC::GetBlockIdx() >= info->qp_num) { AscendC::Trap(); return; }
        const uint64_t index = uint64_t(peer) * info->qp_num + AscendC::GetBlockIdx();
        sq = reinterpret_cast<__gm__ aclshmemi_rdma_sq_ctx*>(info->sq_ptr + index * sizeof(*sq));
        cq = reinterpret_cast<__gm__ aclshmemi_rdma_cq_ctx*>(info->scq_ptr + index * sizeof(*cq));
        if (sq->depth < 2 || (sq->depth & (sq->depth - 1)) || cq->depth < 2 ||
            (cq->depth & (cq->depth - 1)) || (cq->cqe_size && cq->cqe_size < 32)) AscendC::Trap();
    }

    __aicore__ inline void Poll(__gm__ aclshmemi_rdma_sq_ctx* sq,
                               __gm__ aclshmemi_rdma_cq_ctx* cq, uint32_t target)
    {
        // One 64B WRITE WQE and one requested CQE per post. Do not share this
        // routine with sparse-CQE, multi-WQEBB, unsignaled or multi-writer QPs.
        static_assert(offsetof(aclshmemi_hns_1825_cqe_t, owner_id_qpn) == 0 &&
                      offsetof(aclshmemi_hns_1825_cqe_t, op_sr_wqebb) == 4, "CQE ABI changed");
        uint32_t sqTail = aclshmemi_hns_1825_read_u32_gm(sq->tail_addr);
        if (sqTail == target) return;
        uint32_t tail = aclshmemi_hns_1825_read_u32_gm(cq->tail_addr);
        // Earlier users may have compressed CQEs. Preserve their independent
        // SQ/CQ cursors even though every WRITE in this experiment requests one.
        if (uint32_t(target - sqTail) >= cq->depth || uint32_t(target - sqTail) >= sq->depth) {
            AscendC::Trap(); return;
        }
        const auto begin = profile_.DetailClock();
        profile_.Add(CombineProfile::OpsCqPollCalls);
        const uint32_t stride = cq->cqe_size ? cq->cqe_size : 64U;
        const uint32_t oldTail = tail;
        uint64_t loads = 0;
        while (sqTail != target) {
            auto* addr = reinterpret_cast<__gm__ uint64_t*>(cq->buf_addr +
                         uint64_t(tail & (cq->depth - 1)) * stride);
            const uint64_t start = AscendC::GetSystemCycle();
            uint32_t owner = 0, op = 0;
            for (;;) {
                const uint64_t pair = ld_dev(addr, 0);
                ++loads;
                owner = uint32_t(pair); op = uint32_t(pair >> 32);
                const uint32_t type = op >> 27U;
                const bool parity = ((owner >> 31U) & 1U) != uint32_t((tail & cq->depth) == 0U);
                if (type != 0x1fU && parity && (owner & 0xfffffU) == (sq->wqn & 0xfffffU)) break;
                // Bound failed polling to one second on Ascend950. Host stream
                // timeout remains the independent outer guard.
                if (uint64_t(AscendC::GetSystemCycle()) - start > 1000000000ULL) {
                    AscendC::printf("OPS CQ timeout peer-QPN=%u core=%u tail=%u target=%u owner=%x op=%x\n",
                                    sq->wqn, AscendC::GetBlockIdx(), tail, target, owner, op);
                    AscendC::Trap(); return;
                }
            }
            // Also reject unexpected successful opcodes/counters. Never retire
            // a different request because its owner bit happens to match.
            if ((op >> 27U) != 4U || (op & 0xffffU) != (sqTail & 0xffffU)) {
                AscendC::printf("OPS CQ invalid QPN=%u core=%u tail=%u owner=%x op=%x\n",
                                sq->wqn, AscendC::GetBlockIdx(), tail, owner, op);
                AscendC::Trap(); return;
            }
            ++tail;
            ++sqTail;
        }
        aclshmemi_hns_1825_write_u32_gm(cq->tail_addr, tail);
        aclshmemi_hns_1825_write_u32_gm(sq->tail_addr, sqTail);
        // Same big-endian 24-bit CI record as SHMEM, using Hcomm's device store.
        st_dev(aclshmemi_hns_1825_htobe32(tail & 0xffffffU),
               reinterpret_cast<__gm__ uint32_t*>(cq->db_sw_addr), 0);
        profile_.Add(CombineProfile::OpsCqLoads, loads);
        profile_.Add(CombineProfile::OpsCqEntries, uint32_t(tail - oldTail));
        profile_.Elapsed(CombineProfile::OpsCqTicks, begin);
    }
public:
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* p) { profile_.Bind(p); }
    __aicore__ inline void Quiet(uint32_t peer)
    {
        __gm__ aclshmemi_rdma_sq_ctx* sq;
        __gm__ aclshmemi_rdma_cq_ctx* cq;
        Bind(peer, sq, cq);
        Poll(sq, cq, aclshmemi_hns_1825_read_u32_gm(sq->head_addr));
    }
    __aicore__ inline QueueBatch Make(uint32_t peer, AscendC::LocalTensor<uint8_t> storage, uint32_t bytes)
    {
        QueueBatch b;
        Bind(peer, b.sq, b.cq);
        b.capacity = bytes / 64U;
        if (!b.capacity || b.capacity > 10U || (reinterpret_cast<uint64_t>(storage.GetPhyAddr()) & 63U) ||
            b.capacity >= b.sq->depth || b.capacity >= b.cq->depth) AscendC::Trap();
        b.storage = storage.ReinterpretCast<uint32_t>();
        b.head = aclshmemi_hns_1825_read_u32_gm(b.sq->head_addr);
        auto* info = aclshmemi_qp_info_fetch();
        auto* mr = reinterpret_cast<__gm__ aclshmemi_rdma_mem_info*>(info->mem_ptr);
        const uint32_t lkey = mr[aclshmemi_get_my_pe()].lkey;
        b.localBase = reinterpret_cast<uint64_t>(aclshmemi_get_state()->heap_base);
        b.remoteBase = mr[peer].addr;
        b.keys = uint64_t(aclshmemi_hns_1825_htobe32(mr[peer].rkey)) |
                 uint64_t(aclshmemi_hns_1825_htobe32(lkey & 0xffffff00U)) << 32U;
        b.dataKey = uint64_t(aclshmemi_hns_1825_htobe32((lkey & 0x3fffffffU) | 0x80000000U)) << 32U;
        return b;
    }
    __aicore__ inline void Write(QueueBatch& b, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        if (b.count == b.capacity || !bytes || bytes >= 0x80000000U) { AscendC::Trap(); return; }
        const auto begin = profile_.DetailClock();
        const uint32_t index = b.head + b.count;
        auto* w = reinterpret_cast<__ubuf__ uint64_t*>(b.storage.GetPhyAddr()) + b.count * 8U;
        // Fixed single-SGE HNS-1825 WRITE. All eight words are assigned; no
        // per-row zeroing or repeated MR lookup. Hcomm defaults: CQE=1,fence=1.
        const uint16_t bdsl = aclshmemi_hns_1825_htobe16(uint16_t(2U | ((index & 3U) << 12U)));
        const uint64_t len = aclshmemi_hns_1825_htobe32(bytes);
        w[0] = 0x40U | uint64_t((index & b.sq->depth) != 0U) << 7U | 0xa400U |
               uint64_t(bdsl) << 16U | uint64_t(aclshmemi_hns_1825_htobe32(0x10000000U)) << 32U;
        w[1] = 0;
        w[2] = aclshmemi_hns_1825_htobe32(0x64000000U) | len << 32U;
        w[3] = 0;
        w[4] = aclshmemi_hns_1825_htobe64(b.remoteBase + (reinterpret_cast<uint64_t>(dst) - b.localBase));
        w[5] = b.keys;
        w[6] = aclshmemi_hns_1825_htobe64(reinterpret_cast<uint64_t>(src));
        w[7] = len | b.dataKey;
        ++b.count;
        profile_.Elapsed(CombineProfile::WqePrepareTicks, begin);
        profile_.Add(CombineProfile::WqePreparedPackets);
    }
    __aicore__ inline void Commit(QueueBatch& b, AscendC::LocalTensor<uint8_t> scratch)
    {
        using namespace AscendC;
        if (!b.count) return;
        if (aclshmemi_hns_1825_read_u32_gm(b.sq->head_addr) != b.head) { Trap(); return; }
        uint32_t tail = aclshmemi_hns_1825_read_u32_gm(b.sq->tail_addr);
        const uint32_t depth = b.sq->depth < b.cq->depth ? b.sq->depth : b.cq->depth;
        if (uint64_t(uint32_t(b.head - tail)) + b.count + 10U >= depth) {
            profile_.Add(CombineProfile::OpsCapacityPolls);
            Poll(b.sq, b.cq, b.head);
            tail = aclshmemi_hns_1825_read_u32_gm(b.sq->tail_addr);
        }
        if (uint64_t(uint32_t(b.head - tail)) + b.count >= depth) { Trap(); return; }
        const uint32_t slot = b.head & (b.sq->depth - 1U);
        const uint32_t first = b.count < b.sq->depth - slot ? b.count : b.sq->depth - slot;
        GlobalTensor<uint32_t> sq;
        sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(b.sq->buf_addr));
        aclshmemi_hns_1825_sync_func<HardEvent::S_MTE3>(7U);
        DataCopy(sq[slot * 16U], b.storage, first * 16U);
        if (first != b.count) DataCopy(sq, b.storage[first * 16U], (b.count - first) * 16U);
        aclshmemi_roce_hns_1825_write_invalid_wqebb(b.sq->buf_addr, b.sq->depth, b.head + b.count);
        aclshmemi_hns_1825_sync_func<HardEvent::MTE3_S>(7U);
        dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(b.sq->buf_addr) + slot * 64U, first * 64U);
        if (first != b.count) dcci_cachelines(reinterpret_cast<__gm__ uint8_t*>(b.sq->buf_addr), (b.count - first) * 64U);
        auto ub64 = scratch.ReinterpretCast<uint64_t>();
        auto ub32 = scratch.ReinterpretCast<uint32_t>();
        // Pinned SDK encodes the low PI byte in the 2KiB doorbell window and
        // preserves its legacy-firmware fallback and SHMEM PI mirror.
        aclshmemi_roce_ring_sq_doorbell<aclshmemi_rdma_backend_t::HNS_1825>(
            b.sq, b.head + b.count, ub64, ub32, 7U);
        profile_.Add(CombineProfile::OpsBatchCommits);
        if (b.count > maxBatch_) maxBatch_ = b.count;
        profile_.Set(CombineProfile::OpsBatchMax, maxBatch_);
        profile_.Add(CombineProfile::OpsSqCopies, first == b.count ? 1U : 2U);
        b.head += b.count;
        b.count = 0;
    }
};
}  // namespace opsbench
