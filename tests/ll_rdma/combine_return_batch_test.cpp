#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/combine/low_latency/combine_rdma_return.h"
using namespace AscendC;
int main(){
 cores=16;core=0;unsigned cases=0;
 for(unsigned bytes:{512U,4608U,7680U,65536U,131072U})for(unsigned lanes:{1U,2U,8U})
 for(bool deferred:{false,true}) if(!deferred || lanes>1) for(unsigned pattern:{0U,1U,2U}) {
  const unsigned capacity=LlRdma::kCoreBytes/bytes;
  for(unsigned n:{0U,1U,capacity-1,capacity,capacity+1,capacity*2+3}) {
   // The CPU transport encodes peer by +peer*1024 in one backing array.
   // Separate packet ranges by more than the entire peer bias, unlike hardware
   // where different ranks have independent memories.
   const uint64_t stride=bytes+256U*1024U;
   const uint64_t memoryBytes=(uint64_t(n)+1U)*stride+256U*1024U;
   std::vector<uint8_t> local(memoryBytes),remote(memoryBytes),stage(16*LlRdma::kCoreBytes),scratch(256);
   std::vector<std::vector<uint8_t>> ub(lanes,std::vector<uint8_t>(bytes)),saved;
   std::vector<unsigned> targets,remoteTargets;
   Model::local=local.data();Model::remote=remote.data();Model::pending.clear();Model::before={};
   Model::onPost={};Model::onQuiet={};Model::puts=Model::quiets=0;deferredMte3.clear();issuedMte3=0;
   DispatchDedup::CommArgs a{};a.rank=0;a.rankSize=256;a.reserved0=16;a.peerMemBytes=local.size();
   a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
   for(unsigned peer=0;peer<16;++peer)a.peerMems[peer]=local.data()+peer*1024;
   LlTransport::DispatchRdmaBatch batch;batch.Init(&a,bytes);
#ifdef DEEPEP_LL_COMBINE_PROFILE
   std::vector<uint64_t> record(CombineProfile::kWords);
   CombineProfile::DeviceProfile profiler;
   profiler.Init(reinterpret_cast<uint8_t*>(record.data()), 2);
   batch.SetCombineProfile(&profiler);
#endif
   Model::before=[&](const Model::Request& r){
    unsigned i=unsigned((r.dst-remote.data()-r.peer*1024)/stride);
    assert(r.bytes==bytes && r.qp==core && r.peer==targets.at(i));
    assert(std::memcmp(r.src,saved.at(i).data(),bytes)==0);
   };
   for(unsigned lane=0;lane<lanes;++lane)SetFlag<HardEvent::MTE3_V>(lane);
   for(unsigned i=0;i<n;++i){
#ifdef DEEPEP_LL_COMBINE_EARLY_ISSUE
    batch.SubmitPending(); // before next packet's output/Vector work
#endif
    const bool cross=pattern==0 || (pattern==2 && i%3!=1);
    const unsigned peers[]={16,31,32,63,127,255};
    const unsigned peer=cross?peers[i%6]:1+i%15,lane=i%lanes;
    targets.push_back(peer);if(cross)remoteTargets.push_back(peer);
    WaitFlag<HardEvent::MTE3_V>(lane);
    std::fill(ub[lane].begin(),ub[lane].end(),uint8_t(i%251+1));
    saved.push_back(ub[lane]);
    CombineDedup::SendReturnPacket(batch,&a,peer,i*stride,{ub[lane].data()},bytes,deferred);
    if(deferred || !cross)assert(ub[lane]==saved.back());
    SetFlag<HardEvent::MTE3_V>(lane);
   }
   batch.SubmitPending(); // required before M5 can wait for peer messages
   for(unsigned lane=0;lane<lanes;++lane)WaitFlag<HardEvent::MTE3_V>(lane);
   // M5 may repurpose all M4 UB before tail CQ completion. GM is independent.
   for(auto& lane:ub)std::fill(lane.begin(),lane.end(),0xee);
   batch.Flush({scratch.data()});
   assert(!batch.Pending() && !batch.HasStaged() && Model::pending.empty() && deferredMte3.empty());
   unsigned expectedQuiet=0;
   for(unsigned i=0;i<remoteTargets.size();i+=capacity){
    std::set<unsigned> unique;
    for(unsigned j=i;j<remoteTargets.size() && j<i+capacity;++j)unique.insert(remoteTargets[j]);
    expectedQuiet+=unique.size();
   }
   assert(Model::puts==remoteTargets.size() && Model::quiets==expectedQuiet);
#ifdef DEEPEP_LL_COMBINE_PROFILE
   profiler.Finish(0, 256, n);
   assert(record[CombineProfile::RdmaPackets] == remoteTargets.size());
   assert(record[CombineProfile::RdmaQuietCalls] == expectedQuiet);
   assert(record[CombineProfile::RdmaDeferredPackets] == (deferred ? remoteTargets.size() : 0U));
   const uint64_t putMask=(uint64_t(1)<<CombineProfile::RdmaFirstPut)|(uint64_t(1)<<CombineProfile::RdmaLastPut);
   assert((record[CombineProfile::Valid]&putMask)==(remoteTargets.empty()?0U:putMask));
   if (expectedQuiet) assert(record[CombineProfile::RdmaQuietTicks] >= record[CombineProfile::RdmaQuietMaxTicks]);
#endif
   for(unsigned i=0;i<n;++i){
    auto* base=targets[i]>=16?remote.data():local.data();
    assert(std::memcmp(base+targets[i]*1024+i*stride,saved[i].data(),bytes)==0);
   }
   for(auto& row:events)for(bool live:row)assert(!live);
   ++cases;
  }
 }
 std::cout<<"PASS combine return batching "<<cases<<" cases, full stage, 256 slots, high peers, delayed DMA/NIC and M5 UB reuse\n";
}
