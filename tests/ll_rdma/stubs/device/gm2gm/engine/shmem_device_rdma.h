#pragma once
#include <algorithm>
#include <functional>
#include <vector>
namespace Model {
struct Request { uint8_t* dst; uint8_t* src; uint32_t bytes,peer,qp; uint64_t id=0; };
inline thread_local uint8_t *local=nullptr, *remote=nullptr;
inline thread_local std::vector<Request> pending;
inline thread_local std::function<void(const Request&)> before;
// Optional autonomous NIC model. Existing transport tests retain the explicit
// quiet-driven adversarial model when these hooks are unset.
inline thread_local std::function<void(Request&)> onPost;
inline thread_local std::function<void(uint32_t,uint32_t)> onQuiet;
inline thread_local unsigned quiets=0, puts=0, gets=0;
inline uint8_t* translate(uint8_t* p) { return remote + (p-local); }
}
inline void aclshmemx_roce_qp_put_nbi(uint8_t* d,uint8_t* s,uint8_t* scratch,uint32_t n,int pe,uint32_t qp,uint32_t) {
    std::memset(scratch,0xa5,128); // WQE/CQE workspace really destroys its UB contents.
    Model::Request request{Model::translate(d),s,n,uint32_t(pe),qp};
    ++Model::puts;
    if (Model::onPost) Model::onPost(request);
    Model::pending.push_back(request);
}
inline void aclshmemx_roce_qp_quiet(uint32_t pe,uint32_t qp,uint8_t* scratch,uint32_t syncId) {
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(syncId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(syncId);
    std::memset(scratch,0x5a,128); ++Model::quiets;
    if (Model::onQuiet) {
        Model::onQuiet(pe,qp);
        Model::pending.erase(std::remove_if(Model::pending.begin(),Model::pending.end(),
            [=](const Model::Request& r) { return r.peer==pe && r.qp==qp; }),Model::pending.end());
        return;
    }
    // Adversarial completion order within a phase; no cross-QP completion.
    for (size_t i=Model::pending.size(); i--;) {
        auto r=Model::pending[i];
        if (r.peer!=pe || r.qp!=qp) continue;
        if (Model::before) Model::before(r);
        std::lock_guard<std::mutex> guard(AscendC::memoryMutex);
        std::memcpy(r.dst,r.src,r.bytes);
        Model::pending.erase(Model::pending.begin()+i);
    }
}
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
#include "hns_prepare_test_stub.h"
#endif
