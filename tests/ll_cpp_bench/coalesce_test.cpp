// Execute the production pack/unpack code with destructive SDK scratch and
// deferred MTE3 copies. This is a Host protocol model, not an NPU benchmark.
#define DEEPEP_TEST_DEFERRED_MTE3
#include "metadata_stubs/kernel_operator.h"
#include <cassert>
#include <iostream>
#include <map>
#include <set>
namespace AscendC {
template<class T> void Muls(LocalTensor<T> d,LocalTensor<T> s,T x,uint32_t n) {for(uint32_t i=0;i<n;++i)d.p[i]=s.p[i]*x;}
template<class T> void Adds(LocalTensor<T> d,LocalTensor<T> s,T x,uint32_t n) {for(uint32_t i=0;i<n;++i)d.p[i]=s.p[i]+x;}
template<class T> void Gather(LocalTensor<T> d,LocalTensor<T> s,LocalTensor<uint32_t> offsets,uint32_t base,uint32_t n) {
    for(uint32_t i=0;i<n;++i) std::memcpy(d.p+i,reinterpret_cast<uint8_t*>(s.p)+offsets.p[i]+base,sizeof(T));
}
}
#include "kernels/combine/low_latency/combine_coalesce.h"
namespace CombineDedup {
struct CoalescedReturnTestAccess {
    static constexpr uint64_t base=8ULL<<20;
    static void init(CoalescedReturn& c,const DispatchDedup::CommArgs* a,const CombineCoalesce::Layout& l,uint32_t epoch) {
        c.args_=a; c.l_=l; c.base_=base; c.window_=0; c.epoch_=epoch;
    }
    static void send(CoalescedReturn& c,AscendC::LocalTensor<uint8_t> ub,uint32_t worker) {
        c.WaitProducers(ub);
        for(uint32_t b=worker;b<c.l_.buckets;b+=CombineCoalesce::kWorkers) c.PackAndSend(b,ub);
    }
    static void receive(CoalescedReturn& c,AscendC::LocalTensor<uint8_t> ub,uint32_t worker) {
        for(uint32_t r=worker;r<c.l_.world;r+=CombineCoalesce::kWorkers) c.Receive(r,ub);
    }
};
}
using namespace CombineCoalesce;
using namespace AscendC;
using Access=CombineDedup::CoalescedReturnTestAccess;
int main() {
    constexpr bool server=DEEPEP_LL_COMBINE_COALESCE==2, split=DEEPEP_LL_COMBINE_COALESCE_ROWS==1;
    for(uint32_t bs : {1U,2U,32U,48U,64U,72U,256U}) {
        Layout l;
        assert(Make(8,4,bs,3584,8,384,56,28,server,l));
        assert(l.packet==7680 && l.record==7712 && FrameBytes(l,1)==7776);
        assert(l.stage+uint64_t(l.cores)*l.coreRows*l.record<=l.send);
        assert(l.send+uint64_t(l.buckets)*l.frameCapacity<=l.inbox);
        for(uint32_t n : {0U,1U,l.bucketRows}) {
            const uint32_t frames=split&&n?n:1;
            uint64_t end=0;
            for(uint32_t i=0;i<frames;++i) {
                uint64_t off=FrameOffset(l,i,split);
                assert(off>=end);
                end=off+FrameBytes(l,split&&n?1:n);
                assert(end<=l.frameCapacity && end%32==0);
            }
        }
        for(uint32_t src=0;src<8;++src) for(uint32_t dst=0;dst<8;++dst) {
            auto next=Target(l,src,Bucket(l,dst,server),server);
            assert(next/4==dst/4);
            assert(server?next%4==src%4:next==dst);
        }
    }
    Layout rejected;
    assert(!Make(8,4,0,3584,8,384,56,28,server,rejected));
    assert(!Make(4,4,32,3584,8,384,56,28,server,rejected));
    assert(!Make(8,4,65536,16384,32,384,56,28,server,rejected));
    Layout l;
    assert(Make(8,4,3,3584,8,384,16,8,server,l));
    std::vector<std::vector<uint8_t>> memories(8);
    std::vector<DispatchDedup::CommArgs> args(8);
    for(uint32_t r=0;r<8;++r) memories[r].resize(Access::base+l.bytes,0);
    for(uint32_t r=0;r<8;++r) {
        args[r].rank=r; args[r].rankSize=8; args[r].reserved0=4;
        args[r].reservedConfig[0]=LlRdma::kBackend;
        args[r].reservedConfig[1]=16; args[r].reservedAddresses[0]=memories[r].data();
        args[r].peerMemBytes=memories[r].size();
        for(uint32_t p=0;p<8;++p) if(p/4==r/4) args[r].peerMems[p]=memories[p].data();
    }
    for(uint32_t epoch=1;epoch<=4;++epoch) {
        std::vector<std::vector<CombineDedup::CoalescedReturn>> states(8);
        std::map<std::pair<uint32_t,uint32_t>,uint8_t> expected;
        for(uint32_t r=0;r<8;++r) {
            states[r].resize(l.cores);
            for(uint32_t c=0;c<l.cores;++c) Access::init(states[r][c],&args[r],l,epoch);
        }
        cores=l.cores;
        for(uint32_t r=0;r<8;++r) {
            uint32_t ordinal=0;
            // Empty sources, uneven target sizes and rows whose first bytes
            // differ each generation expose old flags and corrupted scratch.
            for(uint32_t peer=0;peer<8;++peer) if(peer/4!=r/4) for(uint32_t t=0;t<3;++t) {
                if((r+peer+t+epoch)%3==0 || (epoch==2 && r==7)) continue;
                const bool partial=(ordinal%2)!=0;
                ++ordinal;
                core=partial?l.cores-l.recv+(ordinal%l.recv):ordinal%l.direct;
                std::vector<uint8_t> packet(l.packet,uint8_t(epoch*31+r*3+t));
                uint32_t slot=t*8+r;
                states[r][core].Enqueue(peer,COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET+
                    uint64_t(slot)*l.packet,{packet.data()});
                expected[{peer,slot}]=uint8_t(epoch*31+r*3+t);
            }
            for(uint32_t c=0;c<l.cores;++c) if(!Worker(l,c)) {
                core=c; std::vector<uint8_t> scratch(256,0);
                states[r][c].ProducerDone({scratch.data()});
                assert(deferredMte3.empty());
            }
        }
        uint32_t wqes=0,expectedWqes=0;
        for(uint32_t r=0;r<8;++r) {
            Model::local=Model::remote=memories[r].data();
            Model::onPost=[&](Model::Request& q) {
                auto offset=q.dst-memories[r].data()-uint64_t(q.peer)*1024U;
                assert(offset>=0 && uint64_t(offset)+q.bytes<=memories[q.peer].size());
                q.dst=memories[q.peer].data()+offset;
            };
            for(uint32_t worker=0;worker<kWorkers;++worker) {
                core=l.direct+worker; std::vector<uint8_t> ub(190*1024);
                unsigned before=Model::puts;
                Access::send(states[r][core],{ub.data()},worker);
                wqes+=Model::puts-before;
                assert(Model::pending.empty() && deferredMte3.empty());
            }
            for(uint32_t b=0;b<l.buckets;++b) {
                uint32_t peer=Target(l,r,b,server);
                if(peer/4==r/4) continue;
                uint32_t count=0;
                for(auto& entry:expected) if(entry.first.second%8==r && Bucket(l,entry.first.first,server)==b) ++count;
                expectedWqes+=split&&count?count:1;
            }
        }
        assert(wqes==expectedWqes);
        for(uint32_t r=0;r<8;++r) for(uint32_t worker=0;worker<kWorkers;++worker) {
            core=l.direct+worker; std::vector<uint8_t> ub(190*1024);
            Access::receive(states[r][core],{ub.data()},worker);
            assert(deferredMte3.empty());
        }
        for(auto& entry:expected) {
            const uint8_t* out=memories[entry.first.first].data()+COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET+
                uint64_t(entry.first.second)*l.packet;
            for(uint32_t i=0;i<l.packet;++i) assert(out[i]==entry.second);
        }
    }
    std::cout << "coalesce routing, generation, empty buckets, WQE counts and byte-exact unpack: PASS\n";
}
