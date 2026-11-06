#include <cassert>
#include <iostream>
#include <set>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/combine/low_latency/combine_rdma_return.h"
#include "kernels/combine/low_latency/combine_core_layout.h"
using namespace AscendC;

int main()
{
    cores=56;core=33;
    const unsigned host=DispatchDedup::kLlPhysicalRanksPerHost;
    unsigned cases=0;
    for(unsigned world:{2U*host,4U*host}) for(unsigned rank:{0U,world-1U})
    for(unsigned dataBytes:{32U,480U,512U,7168U,14336U}) for(bool partial:{false,true}) {
        core = CombineDedup::CombinePhysicalCore(partial ? cores - 8U : 0U, cores);
        const unsigned bytes=(dataBytes+479U)/480U*512U, capacity=LlRdma::kCoreBytes/bytes, n=capacity+5U;
        const uint64_t stride=bytes+world*1024U;
        std::vector<uint8_t> local((n+1U)*stride),remote(local.size()),stage(cores*LlRdma::kCoreBytes);
        std::vector<uint8_t> packet(bytes),low(bytes),scratch(128),quietScratch(128);
        std::vector<unsigned> targets,remoteTargets;
        std::vector<std::vector<uint8_t>> expected;
        Model::local=local.data();Model::remote=remote.data();Model::pending.clear();
        Model::puts=Model::quiets=0;Model::onQuiet={};deferredMte3.clear();issuedMte3=0;
        DispatchDedup::CommArgs a{};a.rank=rank;a.rankSize=world;a.reserved0=host;a.peerMemBytes=local.size();
        a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
        a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
        for(unsigned p=rank/host*host;p<(rank/host+1U)*host;++p)a.peerMems[p]=local.data()+p*1024U;
        CombineDedup::ReturnBatch batch;batch.Init(&a,bytes);
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
        if(partial)batch.SetPartialScratch({scratch.data()});
        constexpr unsigned depth=512;
        std::vector<aclshmemi_rdma_sq_ctx> sq(world*cores);
        std::vector<aclshmemi_rdma_cq_ctx> cq(world*cores);
        std::vector<aclshmemi_rdma_mem_info> mr(world);
        std::vector<std::vector<uint8_t>> rings(world,std::vector<uint8_t>(depth*64U));
        std::vector<uint32_t> heads(world,depth-1),tails(heads);
        for(unsigned p=0;p<world;++p){
            sq[p*cores+core]={uint64_t(rings[p].data()),uint64_t(&heads[p]),uint64_t(&tails[p]),depth};
            cq[p*cores+core].depth=depth;mr[p]={remote.data(),100U+p,200U+p};
        }
        HnsModel::info={uint64_t(sq.data()),uint64_t(cq.data()),uint64_t(mr.data()),cores};
        HnsModel::state={local.data()};HnsModel::rank=rank;HnsModel::builds=HnsModel::published=0;
        HnsModel::heads.clear();
        HnsModel::beforePrepare=[](){assert(events[int(HardEvent::MTE3_S)][7]);assert(!deferredMte3.empty());};
        HnsModel::beforePublish=[](){assert(!events[int(HardEvent::MTE3_S)][7]);assert(deferredMte3.empty());};
        unsigned fallbacks=0, deferred=0;
#endif
#ifdef DEEPEP_LL_COMBINE_PROFILE
        std::vector<uint64_t> record(CombineProfile::kWords*cores);
        CombineProfile::DeviceProfile profile;profile.Init(reinterpret_cast<uint8_t*>(record.data()),2);
        batch.SetCombineProfile(&profile);
#endif
        const auto check=[&](const Model::Request& r){
            const unsigned i=unsigned((r.dst-remote.data()-r.peer*1024U)/stride);
            assert(r.qp==core && r.peer==targets.at(i) && r.bytes==bytes);
            const auto& want=expected.at(i);
            for(unsigned b=0;b<bytes/512U;++b){
                const unsigned take=std::min(480U,dataBytes-b*480U);
                assert(std::memcmp(r.src+b*512U,want.data()+b*512U,take)==0);
                assert(std::memcmp(r.src+b*512U+480U,want.data()+b*512U+480U,32U)==0);
            }
        };
        Model::before=check;Model::onPost=[&](Model::Request& r){assert(deferredMte3.empty());check(r);};
        for(unsigned i=0;i<n;++i){
            const bool cross=i%4U!=1U;
            const unsigned remoteHost=(rank/host+1U+i%(world/host-1U))%(world/host);
            const unsigned peer=(cross?remoteHost:rank/host)*host+i%host;
            targets.push_back(peer);if(cross)remoteTargets.push_back(peer);
            std::fill(packet.begin(),packet.end(),0);std::fill(low.begin(),low.end(),0);
            for(unsigned j=0;j<dataBytes;++j)packet[j/480U*512U+j%480U]=low[j]=uint8_t((j+i)%251U+1U);
            for(unsigned j=0;j<bytes/512U;++j)for(unsigned k=0;k<8U;++k){float one=1;std::memcpy(packet.data()+j*512U+480U+k*4U,&one,4);}
            expected.push_back(packet);
            const unsigned postsBefore=Model::puts;
            const bool defer=(DEEPEP_LL_COMBINE_READY==11 || DEEPEP_LL_COMBINE_READY==15 || DEEPEP_LL_COMBINE_READY==21 || DEEPEP_LL_COMBINE_READY==22 || DEEPEP_LL_COMBINE_READY==23) && partial && cross && i%3U!=2U && i+1U<n && (DEEPEP_LL_COMBINE_READY<21 || DEEPEP_LL_COMBINE_READY==23 || i>0U);
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
            HnsModel::peer=peer;
            // Force admission failure without consuming the live payload event.
            const bool full=partial && cross && i%7U==3U;
            cq[peer*cores+core].depth=full?10U:depth;if(full)++fallbacks;
#endif
#if DEEPEP_LL_COMBINE_READY == 10
            if(partial && cross)batch.EnqueueLinear(peer,i*stride,{reinterpret_cast<uint16_t*>(low.data())},{packet.data()},dataBytes);
            else
#endif
                CombineDedup::SendReturnPacket(batch,&a,peer,i*stride,{packet.data()},bytes,defer);
#if DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
            if(defer) {
                ++deferred;
                assert(batch.HasStaged() && batch.Pending());
                assert(Model::puts==postsBefore && events[int(HardEvent::MTE3_S)][7]);
                assert(!deferredMte3.empty());
                // Work on a disjoint input lane while payload and private WQE
                // remain live. No NIC publication may happen before the wait.
                std::vector<uint8_t> input(64,0x12), prefetch(64);
                DataCopy(LocalTensor<uint8_t>{prefetch.data()},GlobalTensor<uint8_t>{input.data()},64);
                assert(Model::puts==postsBefore);
                batch.SubmitPending();
                assert(Model::puts==postsBefore+1U && !batch.HasStaged());
            }
#endif
            // Retire local MTE as the real output-lane handshake does.
            SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
            assert(!batch.HasStaged());
            std::fill(packet.begin(),packet.end(),0xee);std::fill(low.begin(),low.end(),0xdd);
            std::fill(scratch.begin(),scratch.end(),0xcc); // descriptor is no longer live
        }
        batch.Flush({quietScratch.data()});
        assert(!batch.Pending() && Model::pending.empty() && deferredMte3.empty());
        assert(Model::puts==remoteTargets.size());
        unsigned quiets=0;
        for(unsigned begin=0;begin<remoteTargets.size();begin+=capacity){
            std::set<unsigned> peers;
            for(unsigned i=begin;i<std::min<unsigned>(begin+capacity,remoteTargets.size());++i)peers.insert(remoteTargets[i]);
            quiets+=peers.size();
        }
        assert(Model::quiets==quiets);
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
        assert(HnsModel::builds==HnsModel::published);
        assert(HnsModel::published==(partial?Model::puts-fallbacks:0U));
        if(partial)assert(std::find(HnsModel::heads.begin(),HnsModel::heads.end(),depth)!=HnsModel::heads.end());
#endif
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile.Finish(rank,world,n);auto* r=record.data()+core*CombineProfile::kWords;
        assert(r[CombineProfile::RdmaPackets]==Model::puts);
        assert(r[CombineProfile::RdmaStageWaitCalls]==Model::puts);
#if DEEPEP_LL_COMBINE_READY == 9 || DEEPEP_LL_COMBINE_READY >= 11
        assert(r[CombineProfile::WqePreparedPackets]==HnsModel::published);
        assert(r[CombineProfile::WqeFallbackPackets]==fallbacks);
#if DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
        assert(r[CombineProfile::RdmaDeferredPackets]==deferred);
#endif
#else
        assert(r[CombineProfile::MtePackPackets]==(partial?Model::puts:0U));
#endif
#endif
        for(auto& direction:events)for(bool live:direction)assert(!live);
        ++cases;
    }
    std::cout<<"PASS "<<cases<<" transfer cases: delayed DMA/NIC, MR mapping, SQ wrap/admission, 2/4 hosts, tail packets, GM reuse\n";
}
