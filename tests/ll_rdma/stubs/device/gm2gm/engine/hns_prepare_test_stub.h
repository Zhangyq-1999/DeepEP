#pragma once
#include <cassert>
// HNS API contract model for Combine's adapter, not a hardware WQE encoder.
// Production uses the SDK's encoder/doorbell helpers without redefining them.
enum class aclshmemi_rdma_backend_t { HNS_1825 };
#define ACLSHMEMI_K_RDMA_BACKEND aclshmemi_rdma_backend_t::HNS_1825
enum class aclshmemi_hns_1825_msg_type_t { ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_WRITE };
struct aclshmemi_rdma_sq_ctx { uint64_t buf_addr,head_addr,tail_addr;uint32_t depth; };
struct aclshmemi_rdma_cq_ctx { uint32_t depth; };
struct aclshmemi_rdma_mem_info { uint8_t* addr;uint32_t rkey,lkey; };
struct aclshmemi_rdma_info {uint64_t sq_ptr,scq_ptr,mem_ptr;uint32_t qp_num;};
struct aclshmemi_rdma_send_wr {uint8_t *local_addr,*remote_addr;uint64_t message_len;uint32_t rkey,lkey;};
struct HnsState {uint8_t* heap_base;};
namespace HnsModel {
inline aclshmemi_rdma_info info{};
inline HnsState state{};
inline uint32_t rank=0,peer=0,builds=0,published=0;
inline std::function<void()> beforePrepare, beforePublish;
inline std::vector<uint32_t> heads;
}
inline aclshmemi_rdma_info* aclshmemi_qp_info_fetch(){return &HnsModel::info;}
inline uint32_t aclshmemi_get_my_pe(){return HnsModel::rank;}
inline HnsState* aclshmemi_get_state(){return &HnsModel::state;}
inline uint32_t aclshmemi_hns_1825_read_u32_gm(uint64_t p){return *reinterpret_cast<uint32_t*>(p);}
inline uint8_t* aclshmemi_remote_mr_addr(uint8_t* p,uint8_t* base,uint8_t* remote){return remote+(p-base);}
inline AscendC::LocalTensor<uint32_t> aclshmemi_hns_1825_make_wqe_ub(AscendC::LocalTensor<uint32_t> p,uint32_t){return p;}
inline uint8_t* aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg(uint8_t* raw,uint32_t head,uint32_t depth,bool submit){
    assert(submit); AscendC::LocalTensor<uint32_t> p{reinterpret_cast<uint32_t*>(raw)};
    if(HnsModel::beforePrepare)HnsModel::beforePrepare();
    ++HnsModel::builds;
    std::memset(p.GetPhyAddr(),0,64);p.SetValue(0,head);p.SetValue(1,depth);
    return reinterpret_cast<uint8_t*>(p.GetPhyAddr())+16;
}
inline uint8_t* aclshmemi_roce_hns_1825_fill_wqe_task_seg(uint8_t* p,uint32_t rkey,uint8_t* remote,uint32_t lkey,uint32_t bytes,aclshmemi_hns_1825_msg_type_t,bool submit){
    assert(submit); aclshmemi_rdma_send_wr wr{}; wr.rkey=rkey;wr.remote_addr=remote;wr.lkey=lkey;wr.message_len=bytes;
    std::memcpy(p,&wr.remote_addr,8);std::memcpy(p+8,&wr.message_len,8);
    std::memcpy(p+16,&wr.rkey,4);std::memcpy(p+20,&wr.lkey,4);return p+32;
}
inline uint8_t* aclshmemi_roce_hns_1825_fill_wqe_data_seg(uint8_t* p,uint32_t lkey,uint8_t* local,uint32_t bytes){
    aclshmemi_rdma_send_wr wr{};wr.lkey=lkey;wr.local_addr=local;wr.message_len=bytes;
    std::memcpy(p,&wr.local_addr,8);std::memcpy(p+8,&wr.message_len,8);return p+16;
}
inline uint8_t* aclshmemi_roce_hns_1825_get_send_wqe(aclshmemi_rdma_sq_ctx*& sq,uint32_t i){return reinterpret_cast<uint8_t*>(sq->buf_addr)+i*64;}
inline void aclshmemi_roce_hns_1825_write_invalid_wqebb(uint64_t,uint32_t,uint32_t){
    if(HnsModel::beforePublish)HnsModel::beforePublish();
}
template<class T> void aclshmemi_roce_write_ub_to_gm_with_sync(uint64_t p,AscendC::LocalTensor<T>& ub,uint32_t bytes,uint32_t id){
    using namespace AscendC;
    SetFlag<HardEvent::S_MTE3>(id);WaitFlag<HardEvent::S_MTE3>(id);
    GlobalTensor<T> gm;gm.SetGlobalBuffer(reinterpret_cast<T*>(p));DataCopy(gm,ub,bytes/sizeof(T));
    SetFlag<HardEvent::MTE3_S>(id);WaitFlag<HardEvent::MTE3_S>(id);
}
template<aclshmemi_rdma_backend_t> void aclshmemi_roce_ring_sq_doorbell(aclshmemi_rdma_sq_ctx*& sq,uint32_t head,
    AscendC::LocalTensor<uint64_t>&,AscendC::LocalTensor<uint32_t>&,uint32_t){
    auto* p=aclshmemi_roce_hns_1825_get_send_wqe(sq,(head-1)&(sq->depth-1));
    uint8_t *dst,*src;uint64_t bytes;
    std::memcpy(&dst,p+16,8);std::memcpy(&bytes,p+24,8);std::memcpy(&src,p+48,8);
    uint32_t rkey,lkey;std::memcpy(&rkey,p+32,4);std::memcpy(&lkey,p+36,4);
    if(rkey!=100U+HnsModel::peer || lkey!=200U+HnsModel::rank)throw std::runtime_error("MR key mismatch");
    if(HnsModel::beforePublish)HnsModel::beforePublish();
    ++HnsModel::published;HnsModel::heads.push_back(head);*reinterpret_cast<uint32_t*>(sq->head_addr)=head;
    Model::Request r{dst,src,uint32_t(bytes),HnsModel::peer,AscendC::GetBlockIdx()};
    ++Model::puts;if(Model::onPost)Model::onPost(r);Model::pending.push_back(r);
}
