#pragma once
// Host-only asynchronous NIC model. Wire structs/encoder/doorbell writer are
// extracted from the pinned SDK by test_ops_queue.py, not reimplemented here.
#include "kernel_operator.h"
#include <array>
#include <cassert>
#include <deque>
#include <memory>
#include <vector>
#define GM_ADDR uint8_t*
#define ACLSHMEM_DEVICE inline
constexpr int PIPE_MTE3 = 1;
inline void dcci_cachelines(uint8_t*, uint64_t) {}
inline uint64_t ld_dev(uint64_t*, int);
inline void st_dev(uint32_t, uint32_t*, int);
inline void st_dev(uint64_t, uint64_t*, int);
#include "sdk_wire.hpp"

namespace QueueModel {
constexpr unsigned ranks=64, qps=56, depth=32, heapBytes=24*7680;
inline unsigned myPe=1, doors=0, publicQuiets=0, ciWrites=0, loads=0, copies=0;
inline unsigned inject=0; // 1:error, 2:bad sequence, 3:stale owner, 4:wrong QPN
struct MemoryState { uint8_t* heap_base; };
inline std::array<uint8_t,heapBytes> local;
inline MemoryState state{local.data()};
inline aclshmemi_rdma_info info;
inline std::array<aclshmemi_rdma_sq_ctx,ranks*qps> sqs;
inline std::array<aclshmemi_rdma_cq_ctx,ranks*qps> cqs;
inline std::array<aclshmemi_rdma_mem_info,ranks> mrs;
struct Request { uint8_t* dst; uint8_t* src; uint32_t bytes,index; };
struct Ring {
    alignas(64) std::array<uint8_t,depth*64> sq{}, cq{};
    alignas(2048) std::array<uint64_t,256> db{};
    std::array<uint8_t,heapBytes> remote{};
    uint32_t head=0,tail=0,cqTail=0,cqProduced=0,cqDb=0,swDb=0,published=0;
    std::deque<Request> pending;
    std::vector<unsigned> batches;
    unsigned peer,core;
    Ring(unsigned p,unsigned c,uint32_t start,uint32_t skipped):head(start),tail(start),cqTail(start-skipped),cqProduced(start-skipped),published(start),peer(p),core(c) {
        auto& s=sqs[p*qps+c]; auto& q=cqs[p*qps+c];
        s={}; q={};
        s.wqn=7+p*qps+c; s.buf_addr=(uint64_t)sq.data(); s.depth=depth;
        s.wqe_size=256; // deliberately not the 64B WQEBB stride
        s.head_addr=(uint64_t)&head; s.tail_addr=(uint64_t)&tail;
        s.db_addr=(uint64_t)db.data(); s.db_sw_addr=(uint64_t)&swDb;
        s.db_cos=7; s.mtu_shift=4;
        q.buf_addr=(uint64_t)cq.data(); q.depth=depth; q.cqe_size=64;
        q.tail_addr=(uint64_t)&cqTail; q.db_sw_addr=(uint64_t)&cqDb;
        if(!mrs[p].addr) mrs[p]={heapBytes,(uint64_t)remote.data(),0x12345678U+p,0xaabb0000U+p};
        for(unsigned i=0;i<depth;++i) {
            auto* e=(aclshmemi_hns_1825_cqe_t*)(cq.data()+i*64);
            e->op_sr_wqebb=0x1fU<<27;
        }
    }
    void publish(uint32_t next) {
        assert(AscendC::deferredMte3.empty()); // SQ complete before DB
        assert(uint32_t(next-tail)<depth && uint32_t(next-published)<=10);
        batches.push_back(next-published); ++doors;
        assert(sq[(next&(depth-1))*64] == ((next&depth)?0x7f:0xff));
        while(published!=next) {
            auto* w=sq.data()+(published&(depth-1))*64;
            auto* ctrl=(aclshmemi_hns_1825_wqe_ctrl_seg_t*)w;
            auto* t=(aclshmemi_hns_1825_wqe_rdma_task_seg_t*)(w+16);
            auto* d=(aclshmemi_hns_1825_wqe_data_seg_t*)(w+48);
            assert(ctrl->owner_sl==(0x40U|((published&depth)?0x80U:0)));
            assert(aclshmemi_hns_1825_htobe16(ctrl->wf_bdsl)==(2U|((published&3)<<12)));
            auto opcode=aclshmemi_hns_1825_htobe32(t->com_tsk.value);
            assert(opcode==0x24000000U || opcode==0x64000000U);
            assert(aclshmemi_hns_1825_htobe32(t->rkey)==mrs[peer].rkey);
            assert(aclshmemi_hns_1825_htobe32(d->le_key)==((mrs[myPe].lkey&0x3fffffff)|0x80000000U));
            auto* dst=(uint8_t*)aclshmemi_hns_1825_htobe64(t->va);
            auto* src=(uint8_t*)aclshmemi_hns_1825_htobe64(d->buf_addr);
            auto bytes=aclshmemi_hns_1825_htobe32(t->data_len);
            assert(bytes==aclshmemi_hns_1825_htobe32(d->r_len));
            assert(dst>=remote.data() && dst+bytes<=remote.data()+remote.size());
            assert(src>=local.data() && src+bytes<=local.data()+local.size());
            pending.push_back({dst,src,bytes,published++});
        }
    }
    void progress() {
        while(!pending.empty()) {
            auto r=pending.front(); pending.pop_front();
            std::memcpy(r.dst,r.src,r.bytes);
            auto* e=(aclshmemi_hns_1825_cqe_t*)(cq.data()+(cqProduced&(depth-1))*64);
            e->owner_id_qpn=((cqProduced++&depth)?0x80000000U:0U)|sqs[peer*qps+core].wqn;
            e->op_sr_wqebb=(4U<<27)|(r.index&0xffffU);
            if(inject==1) e->op_sr_wqebb=0x1eU<<27;
            if(inject==2) e->op_sr_wqebb^=1U;
            if(inject==3) e->owner_id_qpn^=0x80000000U;
            if(inject==4) e->owner_id_qpn^=1U;
        }
    }
};
inline std::vector<std::unique_ptr<Ring>> rings;
inline Ring& add(unsigned peer,unsigned core,uint32_t start=0,uint32_t skipped=0) {
    rings.push_back(std::make_unique<Ring>(peer,core,start,skipped)); return *rings.back();
}
inline Ring& find(unsigned peer,unsigned core) {
    for(auto& r:rings) if(r->peer==peer && r->core==core) return *r;
    throw std::runtime_error("unbound peer/QP");
}
inline void reset() {
    rings.clear(); sqs={}; cqs={}; mrs={}; doors=publicQuiets=ciWrites=loads=inject=0;
    info={qps,(uint64_t)sqs.data(),0,(uint64_t)cqs.data(),0,(uint64_t)mrs.data()};
    mrs[myPe]={heapBytes,(uint64_t)local.data(),0x12345678U,0x87654321U};
    AscendC::cycles=100; AscendC::core=55;
}
}
inline auto* aclshmemi_qp_info_fetch() { return &QueueModel::info; }
inline auto* aclshmemi_get_state() { return &QueueModel::state; }
inline unsigned aclshmemi_get_my_pe() { return QueueModel::myPe; }
constexpr auto ACLSHMEMI_K_RDMA_BACKEND=aclshmemi_rdma_backend_t::HNS_1825;
// Only LocalTensor view construction/address translation is modeled. WQE
// encoding, publication and the new SDK signatures are exercised below.
inline AscendC::LocalTensor<uint32_t> aclshmemi_hns_1825_make_wqe_ub(AscendC::LocalTensor<uint32_t>& ub,uint32_t n) {
    assert(n==64); return ub;
}
inline uint8_t* aclshmemi_remote_mr_addr(uint8_t* dst,uint8_t* base,uint64_t remote) {
    return reinterpret_cast<uint8_t*>(remote+(dst-base));
}
inline uint64_t ld_dev(uint64_t* p,int) {
    using namespace QueueModel;
    ++loads; AscendC::cycles+=250000000; // deterministic timeout in model
    for(auto& r:rings) if((uint8_t*)p>=r->cq.data() && (uint8_t*)p<r->cq.data()+r->cq.size()) {
        if(loads%2==0) r->progress(); // at least one missed load
        return *p;
    }
    throw std::runtime_error("load outside CQ");
}
inline void st_dev(uint32_t v,uint32_t* p,int) { *p=v; ++QueueModel::ciWrites; }
inline void st_dev(uint64_t v,uint64_t* p,int) {
    using namespace QueueModel;
    for(auto& r:rings) if(p>=r->db.data() && p<r->db.data()+r->db.size()) {
        assert((v&0xfffffU)==sqs[r->peer*qps+r->core].wqn);
        assert(((v>>27)&31)==21 && ((v>>32)&255)==((r->head>>8)&255));
        assert(unsigned(p-r->db.data())==(r->head&255));
        r->publish(r->head); return;
    }
    throw std::runtime_error("store outside doorbell");
}
inline void aclshmemx_roce_qp_put_nbi(uint8_t* dst,uint8_t* src,uint8_t*,uint32_t bytes,unsigned peer,unsigned core,unsigned) {
    using namespace QueueModel;
    auto& r=find(peer,core); auto* s=&sqs[peer*qps+core];
    assert(uint32_t(r.head-r.tail)+1<depth);
    alignas(64) uint32_t words[16]; AscendC::LocalTensor<uint32_t> ub{words};
    aclshmemi_rdma_send_wr wr{}; wr.local_addr=src; wr.remote_addr=r.remote.data()+(dst-local.data());
    wr.message_len=bytes; wr.lkey=mrs[myPe].lkey; wr.rkey=mrs[peer].rkey;
    auto* t=aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg(reinterpret_cast<uint8_t*>(ub.GetPhyAddr()),r.head,depth,true);
    auto* d=aclshmemi_roce_hns_1825_fill_wqe_task_seg(t,wr.rkey,wr.remote_addr,wr.lkey,uint32_t(wr.message_len),aclshmemi_hns_1825_msg_type_t::ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_WRITE,true);
    aclshmemi_roce_hns_1825_fill_wqe_data_seg(d,wr.lkey,wr.local_addr,uint32_t(wr.message_len));
    std::memcpy(r.sq.data()+(r.head&(depth-1))*64,words,64);
    aclshmemi_roce_hns_1825_write_invalid_wqebb(s->buf_addr,s->depth,++r.head);
    r.publish(r.head);
}
inline void aclshmemx_roce_qp_quiet(unsigned peer,unsigned core,uint8_t*,unsigned) {
    auto& r=QueueModel::find(peer,core); r.progress(); r.tail=r.head; r.cqTail=r.cqProduced; ++QueueModel::publicQuiets;
}
