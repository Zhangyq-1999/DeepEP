#include "queue_test_model.hpp"
#include "transport.hpp"
#include "prepared.hpp"
#include <iostream>

using namespace QueueModel;

static void checkWire(const opsbench::QueueBatch& b,unsigned row,uint8_t* dst,uint8_t* src,unsigned bytes)
{
    // Independent oracle: pinned SDK encoder, with only Hcomm's fence default.
    alignas(64) uint32_t expected[16];
    AscendC::LocalTensor<uint32_t> ub{expected};
    aclshmemi_rdma_send_wr wr{};
    wr.local_addr=src; wr.remote_addr=(uint8_t*)(b.remoteBase+(dst-local.data()));
    wr.message_len=bytes; wr.lkey=mrs[myPe].lkey;
    wr.rkey=aclshmemi_hns_1825_htobe32(uint32_t(b.keys));
    auto* t=aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg(reinterpret_cast<uint8_t*>(ub.GetPhyAddr()),b.head+row,b.sq->depth,true);
    auto* d=aclshmemi_roce_hns_1825_fill_wqe_task_seg(t,wr.rkey,wr.remote_addr,wr.lkey,uint32_t(wr.message_len),aclshmemi_hns_1825_msg_type_t::ACLSHMEMI_HNS_1825_MSG_OPCODE_RDMA_WRITE,true);
    aclshmemi_roce_hns_1825_fill_wqe_data_seg(d,wr.lkey,wr.local_addr,uint32_t(wr.message_len));
    expected[4] |= aclshmemi_hns_1825_htobe32(1U<<30);
    assert(std::memcmp(expected,b.storage.p+row*16,64)==0);
}

int main()
{
    // Existing DeepEP's prepare/publish adapter must still compile and publish
    // the real SDK encoding after the IPv6 dependency's internal API changes.
    reset();
    auto& preparedRing=add(17,55,31,13);
    alignas(64) uint8_t preparedScratch[128]{};
    CombineDedup::PreparedWrite prepared;
    assert(prepared.Prepare(17,local.data(),local.data(),512,{preparedScratch}));
    assert(doors==0 && preparedRing.head==31);
    std::memset(local.data(),0x37,512);
    prepared.Publish();
    assert(doors==1 && preparedRing.head==32);
    std::memset(preparedScratch,0xcc,sizeof(preparedScratch));
    opsbench::RoceQueue preparedQueue;
    preparedQueue.Quiet(17);
    assert(preparedRing.tail==32 && preparedRing.cqTail==19);
    assert(std::memcmp(local.data(),preparedRing.remote.data(),512)==0);
    preparedRing.head=preparedRing.tail+22;
    assert(!prepared.Prepare(17,local.data(),local.data(),512,{preparedScratch}));

    // Independent QP for every physical core; peer IDs cover three remote
    // servers in EP64. Nonzero/high PI validates both owner and counter wrap.
    for(uint32_t skipped:{0U,13U}) for(uint32_t start:{0U,29U,65533U,0xfffffffdU}) {
        reset();
        for(unsigned p:{17U,33U,49U}) add(p,55,start,skipped);
        auto& untouched=add(17,54,start);
        for(unsigned generation=1;generation<=6;++generation) {
            opsbench::Transport tx;
            alignas(64) uint8_t scratch[128]{},storage[640]{},fresh[128]{};
#ifdef DEEPEP_LL_COMBINE_PROFILE
            std::vector<uint64_t> records(56*CombineProfile::kWords);
            CombineProfile::DeviceProfile profile;
            profile.Init((uint8_t*)records.data(),2); tx.SetProfile(&profile);
#endif
            tx.Init({scratch},sizeof(scratch)); tx.EnableDeferred(true);
            for(unsigned i=0;i<local.size();++i) local[i]=uint8_t(generation+i/7680);
            for(unsigned p:{17U,33U,49U}) {
                auto batch=tx.MakeBatchHandle(p,{storage},sizeof(storage),local.data(),local.data());
                const unsigned before=doors;
                for(unsigned row=0;row<24;++row) {
                    auto* ptr=local.data()+row*7680;
                    tx.WriteNbi(batch,ptr,ptr,7680);
#if DEEPEP_LL_OPS_POLICY == 4
                    checkWire(batch.queue,batch.queue.count-1,ptr,ptr,7680);
                    if(row<10) assert(doors==before);
#endif
                    if(row%10==9 || row==23) {
                        tx.BatchCommit(batch);
                        assert(AscendC::deferredMte3.empty());
                        // UB is reusable while the NIC still owns source GM.
                        std::memset(storage,0xcd,sizeof(storage));
                        std::memset(scratch,0xab,sizeof(scratch));
                    }
                }
#if DEEPEP_LL_OPS_POLICY == 4
                assert(doors-before==3);
#else
                assert(doors-before==24);
#endif
            }
            assert(tx.HasPending());
            tx.Retire({fresh});
            assert(!tx.HasPending() && publicQuiets==0);
            for(unsigned p:{17U,33U,49U}) {
                auto& r=find(p,55);
                assert(r.pending.empty() && r.head==r.tail && r.tail-skipped==r.cqTail);
                assert(aclshmemi_hns_1825_htobe32(r.cqDb)==(r.cqTail&0xffffff));
                assert(std::memcmp(local.data(),r.remote.data(),heapBytes)==0);
            }
            assert(untouched.head==start && untouched.tail==start && untouched.batches.empty());
#ifdef DEEPEP_LL_COMBINE_PROFILE
            profile.Finish(1,64,2);
            using namespace CombineProfile;
            auto* r=records.data()+55*kWords;
            assert(r[RdmaPackets]==72 && r[OpsCqEntries]==72 && r[OpsCqLoads]>=72);
            assert(r[OpsRetiredPeers]==3 && r[OpsCqPollCalls]>=3);
            assert(r[RdmaFirstPut]<=r[RdmaLastPut]);
#if DEEPEP_LL_OPS_POLICY == 4
            assert(r[WqePreparedPackets]==72 && r[OpsBatchCommits]==9 && r[OpsBatchMax]==10);
            assert(r[OpsSqCopies]>=9 && r[OpsCapacityPolls]>0);
#endif
#endif
        }
        // Larger shape fallback uses unchanged public put + per-batch quiet.
        opsbench::Transport fallback; alignas(64) uint8_t scratch[640]{};
        fallback.Init({scratch},128); fallback.EnableDeferred(false);
        auto b=fallback.MakeBatchHandle(17,{scratch},640,local.data(),local.data());
        fallback.WriteNbi(b,local.data(),local.data(),512); fallback.BatchCommit(b);
        assert(publicQuiets==1 && !fallback.HasPending());
    }
    // Wrong/error/stale CQEs must never retire an outstanding send. Timeout
    // is model-clock driven; these do not wait in real time.
    for(unsigned fault:{1U,2U,3U,4U}) {
        reset(); auto& r=add(17,55); inject=fault;
        opsbench::Transport tx; alignas(64) uint8_t scratch[128]{},storage[640]{};
        tx.Init({scratch},128); tx.EnableDeferred(true);
        auto b=tx.MakeBatchHandle(17,{storage},640,local.data(),local.data());
        tx.WriteNbi(b,local.data(),local.data(),512); tx.BatchCommit(b);
        bool trapped=false;
        try { tx.Retire({scratch}); } catch(const std::runtime_error&) { trapped=true; }
        assert(trapped && r.head==1 && r.tail==0 && r.cqTail==0 && ciWrites==0);
    }
    std::cout << "ops queue: SDK wire/DB oracle, asynchronous completion, wrap, capacity, EP64 peers, fallback passed\n";
}
