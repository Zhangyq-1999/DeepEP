"""Exercise the production SIMT publisher and independent consumer rows on CPU."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", [1, 2])
def test_notifications_cover_every_consumer_and_reuse_banks(tmp_path, mode):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU checks")
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h").read_text(encoding="utf-8")
    start = text.index("    __aicore__ inline void PublishReadySimt(")
    method = text[start:text.index("\n#endif", start)]
    start = text.index("    __aicore__ inline void WaitFlags(")
    wait = text[start:text.index("\n#ifdef DEEPEP_LL_COMBINE_PROFILE\n    __aicore__ inline void InspectReadySnapshot", start)]
    source = tmp_path / "notify.cpp"
    source.write_text(r'''
#include "simt_test_stub.h"
#include "kernel_operator.h"
#include "kernels/combine/low_latency/combine_notify.h"
#include "kernels/combine/low_latency/combine_profile_layout.h"
#include <cassert>
#include <algorithm>
#include <numeric>
#include <vector>
using namespace AscendC;
using namespace CombineDedup;
struct Args { unsigned rank; unsigned char* peerMems[16]; };
namespace LlTransport { bool RemoteWrite(const Args*,unsigned) {return false;} }
struct Profile {
 unsigned calls=0,misses=0;
 uint64_t DetailClock(){return 0;}
 void Add(CombineProfile::Word w,unsigned n=1) {
   if(w==CombineProfile::ReadyPollCalls)calls+=n;
   if(w==CombineProfile::ReadyPollMisses)misses+=n;
 }
 template<class... T> void Elapsed(T...) {}
 template<class... T> void MaxElapsed(T...) {}
 template<class... T> void Mark(T...) {}
};
inline float* missing=nullptr;
void PauseAfterFlagMiss(Profile& p){assert(p.misses==1 && missing);*missing=1.0f;}
namespace AscendC {
template<class T> void ReduceSum(LocalTensor<T> out,LocalTensor<T> in,LocalTensor<T>,int32_t n) {
 out.p[0]=std::accumulate(in.p,in.p+n,T(0));
}
template<class T> void ReduceSum(LocalTensor<T> out,LocalTensor<T> in,LocalTensor<T>,unsigned,unsigned n,unsigned) {
 out.p[0]=0;for(unsigned i=0;i<n;++i)out.p[0]+=in.p[i*8];
}
}
template<unsigned Group> struct Publisher {
 Args* args_; unsigned core_,notifyBegin_,recvCores_; uint64_t control_;
 LocalTensor<float> notifyUb_; Profile profile_;
 template<HardEvent E> void Sync() {}
''' + method + r'''
};
template<unsigned Group> struct Reader {
 unsigned coreNum_;Profile profile_;
 template<HardEvent E> void Sync() {}
''' + wait + r'''
};
template<unsigned Group> void check(unsigned cores,unsigned recv) {
 constexpr unsigned marker=0x3f800000U;
 constexpr unsigned rowBytes=DEEPEP_LL_COMBINE_NOTIFY==2?kCoreReadyRowBytes:512;
 constexpr unsigned table=DEEPEP_LL_COMBINE_NOTIFY==2?
   COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CORE_READY_OFFSET:
   COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET;
 const unsigned bankWords=(table+recv*rowBytes+32)/4;
 std::vector<std::vector<unsigned>> memory(Group,std::vector<unsigned>(bankWords*2,0));
 Args args{};
 for(unsigned rank=0;rank<Group;++rank) args.peerMems[rank]=reinterpret_cast<unsigned char*>(memory[rank].data());
 alignas(32) float scratch[288]{};
 Publisher<Group> p{&args,0,cores-8,recv,0,{scratch},{}};
 const unsigned expected=DEEPEP_LL_COMBINE_NOTIFY==2?Group*cores:Group;
 auto checkRow=[&](unsigned bank,unsigned peer,unsigned consumer,unsigned ready) {
   auto begin=memory[peer].begin()+bank*bankWords+table/4+consumer*rowBytes/4;
   assert(std::count(begin,begin+rowBytes/4,marker)==ready);
   for(auto i=begin;i!=begin+rowBytes/4;++i) assert(*i==0 || *i==marker);
 };
 for(unsigned iteration=0;iteration<6;++iteration) {
   const unsigned bank=iteration%2;
   p.control_=bank*bankWords*4;
   for(unsigned peer=0;peer<Group;++peer)for(unsigned c=0;c<recv;++c)checkRow(bank,peer,c,0);
   // Delay one producer rank/core, including a producer that would own no data.
   for(unsigned rank=0;rank<Group;++rank) {
     args.rank=rank;
     for(unsigned core=0;core<cores;++core) {
       p.core_=core;
       if constexpr(DEEPEP_LL_COMBINE_NOTIFY==2) {
         if(rank==Group-1 && core==cores-1)continue;
         p.PublishReadySimt(true);
       } else if(core>=cores-8 && rank!=Group-1) p.PublishReadySimt(false);
     }
   }
   for(unsigned peer=0;peer<Group;++peer)for(unsigned c=0;c<recv;++c)
     checkRow(bank,peer,c,expected-1);
   args.rank=Group-1;
   if constexpr(DEEPEP_LL_COMBINE_NOTIFY==2) {p.core_=cores-1;p.PublishReadySimt(true);}
   else for(unsigned c=cores-8;c<cores;++c){p.core_=c;p.PublishReadySimt(false);}
   // Execute the production Vector polling loop against one delayed slot.
   auto first=memory[0].data()+bank*bankWords+table/4;
   const unsigned slot=DEEPEP_LL_COMBINE_NOTIFY==2?(Group-1)*kCoreReadyPeerWords+cores-1:(Group-1)*8;
   first[slot]=0;missing=reinterpret_cast<float*>(first+slot);
   std::vector<float> input(rowBytes/4+64),work(64),sum(8);
   Reader<Group> reader{cores,{}};
   reader.WaitFlags({reinterpret_cast<float*>(first)},{input.data()},{work.data()},{sum.data()},
       DEEPEP_LL_COMBINE_NOTIFY==2?Group*kCoreReadyPeerWords:Group,true,DEEPEP_LL_COMBINE_NOTIFY==2);
   assert(reader.profile_.calls==2 && reader.profile_.misses==1);
   for(unsigned peer=0;peer<Group;++peer) {
     for(unsigned c=0;c<recv;++c)checkRow(bank,peer,c,expected);
     // Each reader may clear its row while other readers are still consuming.
     for(unsigned c=0;c<recv;++c) {
       auto first=memory[peer].begin()+bank*bankWords+table/4+c*rowBytes/4;
       std::fill(first,first+rowBytes/4,0);
       for(unsigned later=c+1;later<recv;++later)checkRow(bank,peer,later,expected);
     }
     // No stores outside the selected bank/table, including the tail guard.
     assert(std::all_of(memory[peer].begin(),memory[peer].end(),[](unsigned v){return v==0;}));
   }
 }
}
int main(){for(unsigned cores:{16U,56U,72U})for(unsigned recv:{8U,28U,36U})if(recv<cores){
 check<2>(cores,recv);check<4>(cores,recv);check<16>(cores,recv);
}}
''', encoding="utf-8")
    exe = tmp_path / "notify.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG", f"-DDEEPEP_LL_COMBINE_NOTIFY={mode}",
                    "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(source), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=30)


@pytest.mark.parametrize("mode", ["1", "2"])
def test_original_rejects_notification(mode):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "bash"
    proc = subprocess.run([bash, str(ROOT / "tests/ll_cpp_bench/build.sh"),
                           "--variant", "base", "--notify", mode], capture_output=True, text=True, timeout=15)
    assert proc.returncode and "Use --variant opt" in proc.stderr
