#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/combine/low_latency/combine_rdma_return.h"
using namespace AscendC;

int main()
{
    cores=56; core=0;
    const unsigned host=DispatchDedup::kLlPhysicalRanksPerHost;
    unsigned cases=0, paired=0;
    for(unsigned world:{2U*host,4U*host}) for(unsigned rank:{0U,world-1U})
    for(unsigned bytes:{512U,7680U,65536U,131072U}) for(unsigned lanes:{1U,2U,8U})
    for(unsigned pattern:{0U,1U,2U}) for(bool m4:{false,true}) {
        const unsigned capacity=LlRdma::kCoreBytes/bytes;
        const unsigned n=capacity+5U;
        const uint64_t stride=bytes+world*1024U;
        std::vector<uint8_t> local((n+1U)*stride),remote(local.size()),stage(cores*LlRdma::kCoreBytes),scratch(256U);
        std::vector<std::vector<uint8_t>> ub(lanes,std::vector<uint8_t>(bytes)),expected;
        std::vector<unsigned> targets,remoteTargets,postOrder;
        Model::local=local.data();Model::remote=remote.data();Model::pending.clear();
        Model::onQuiet={};Model::puts=Model::quiets=0;deferredMte3.clear();issuedMte3=0;
        DispatchDedup::CommArgs a{};a.rank=rank;a.rankSize=world;a.reserved0=host;a.peerMemBytes=local.size();
        a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
        a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
        for(unsigned peer=rank/host*host;peer<(rank/host+1U)*host;++peer)a.peerMems[peer]=local.data()+peer*1024U;
        CombineDedup::ReturnBatch batch;batch.Init(&a,bytes);
#ifdef DEEPEP_LL_COMBINE_PROFILE
        std::vector<uint64_t> record(CombineProfile::kWords);
        CombineProfile::DeviceProfile profile;profile.Init(reinterpret_cast<uint8_t*>(record.data()),2);
        batch.SetCombineProfile(&profile);
#endif
        const auto check=[&](const Model::Request& r) {
            const unsigned i=unsigned((r.dst-remote.data()-r.peer*1024U)/stride);
            assert(r.qp==core && r.peer==targets.at(i) && r.bytes==bytes);
            assert(std::memcmp(r.src,expected.at(i).data(),bytes)==0);
        };
        Model::before=check; // NIC deliberately reads GM only during Quiet.
        Model::onPost=[&](Model::Request& r) {check(r);postOrder.push_back(unsigned((r.dst-remote.data()-r.peer*1024U)/stride));};
        for(unsigned lane=0;lane<lanes;++lane)SetFlag<HardEvent::MTE3_V>(lane);
        for(unsigned i=0;i<n;++i) {
            const bool cross=pattern==0 || (pattern==2 && i%3!=1);
            const unsigned other=((rank/host+1U+i%(world/host-1U))%(world/host))*host;
            const unsigned peer=cross?other+i%host:rank/host*host+i%host, lane=i%lanes;
            targets.push_back(peer);if(cross)remoteTargets.push_back(peer);
            WaitFlag<HardEvent::MTE3_V>(lane);
            std::fill(ub[lane].begin(),ub[lane].end(),uint8_t(i%251U+1U));expected.push_back(ub[lane]);
            // End every three-task runnable batch before any dependency wait.
            const bool defer=lanes>1U && i+1U<n && i%3U!=2U;
            if(m4 && cross)batch.Enqueue(peer,i*stride,{ub[lane].data()},defer);
            else CombineDedup::SendReturnPacket(batch,&a,peer,i*stride,{ub[lane].data()},bytes,false);
            if(!defer)assert(!batch.HasStaged());
            SetFlag<HardEvent::MTE3_V>(lane);
        }
        assert(!batch.HasStaged()); // No unposted work when M5 starts polling.
        for(unsigned lane=0;lane<lanes;++lane)WaitFlag<HardEvent::MTE3_V>(lane);
        for(auto& lane:ub)std::fill(lane.begin(),lane.end(),0xee); // M5 reuses UB.
        batch.Flush({scratch.data()});
        assert(!batch.Pending() && Model::pending.empty() && deferredMte3.empty());
        unsigned quiets=0;
        for(unsigned begin=0;begin<remoteTargets.size();begin+=capacity) {
            std::set<unsigned> peers;
            for(unsigned i=begin;i<remoteTargets.size() && i<begin+capacity;++i)peers.insert(remoteTargets[i]);
            quiets+=peers.size();
        }
        assert(Model::puts==remoteTargets.size() && Model::quiets==quiets);
        assert(std::is_sorted(postOrder.begin(),postOrder.end()));
        for(unsigned i=0;i<n;++i) {
            const auto* base=LlTransport::RemoteWrite(&a,targets[i])?remote.data():local.data();
            assert(std::memcmp(base+targets[i]*1024U+i*stride,expected[i].data(),bytes)==0);
        }
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile.Finish(rank,world,n);
        assert(record[CombineProfile::RdmaPackets]==Model::puts);
        assert(record[CombineProfile::RdmaStageWaitCalls]+record[CombineProfile::RdmaStagePairGroups]==Model::puts);
        if(!m4 || lanes==1U || capacity==1U)assert(!record[CombineProfile::RdmaStagePairGroups]);
        paired+=unsigned(record[CombineProfile::RdmaStagePairGroups]);
#endif
        for(auto& direction:events)for(bool live:direction)assert(!live);
        ++cases;
    }
#ifdef DEEPEP_LL_COMBINE_PROFILE
    assert(paired>0U);
#endif
    std::cout<<"PASS "<<cases<<" paired staging cases; delayed DMA/NIC, same/mixed peers, four hosts, ring wrap, tail drain\n";
}
