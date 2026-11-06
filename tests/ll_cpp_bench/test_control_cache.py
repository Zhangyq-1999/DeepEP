"""Execute production control access/publish bodies; hardware coherence is not simulated."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", [12, 18, 19, 20])
@pytest.mark.parametrize("recv", [28, 32])
def test_control_views_and_all_notify_destinations(tmp_path, mode, recv):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for control publication checks")
    kernel = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h").read_text(encoding="utf-8")
    control = kernel[kernel.index("    __aicore__ inline GlobalTensor<float> Control("):kernel.index("    __aicore__ inline void SelectRows(")]
    notify = kernel[kernel.index("    __aicore__ inline void NotifyServerCard()"):kernel.index("    __aicore__ inline void WaitServerReady()")]
    ready = (ROOT / "csrc/kernels/combine/low_latency/combine_ready.h").read_text(encoding="utf-8")
    tensor = ready[ready.index("    __aicore__ inline AscendC::GlobalTensor<uint32_t> Tensor("):ready.index("    __aicore__ inline uint64_t ReadMask(")]
    source = r'''
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "kernels/combine/low_latency/combine_ready_layout.h"
#include "kernels/combine/low_latency/combine_profile_layout.h"
#include "kernels/combine/low_latency/ep_memory_server_dedup_buffer_layout.h"
#define __aicore__
#define __gm__
using namespace CombineDedup;
namespace AscendC {
enum class CacheMode {CACHE_MODE_DISABLE};
enum class HardEvent {V_S,V_MTE3,S_MTE3};
template<class T> struct GlobalTensor {
    T* p=nullptr;bool cached=true;
    void SetGlobalBuffer(T* q){p=q;}
    void SetL2CacheHint(CacheMode){cached=false;}
};
template<class T> struct LocalTensor {
    T* p;
    void SetValue(unsigned i,T v){p[i]=v;}
    template<class U> LocalTensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p)};}
};
struct Copy {uint16_t count;uint32_t bytes,srcStride,dstStride,reserved;};
unsigned controls=0;
template<class T> void DataCopyPad(GlobalTensor<T> d,LocalTensor<T> s,Copy c) {
    assert(d.cached==kCombineControlL2);++controls;
    for(unsigned i=0;i<c.count;++i)
        std::memcpy(reinterpret_cast<char*>(d.p)+i*(c.bytes+c.dstStride),
                    reinterpret_cast<char*>(s.p)+i*(c.bytes+c.srcStride),c.bytes);
}
template<class T> void Duplicate(LocalTensor<T> d,T x,unsigned n){std::fill_n(d.p,n,x);}
void Trap(){throw std::runtime_error("invalid host");}
}
using namespace AscendC;
struct Args {unsigned rank=0,host=0;std::vector<uint8_t*> peerMems;};
namespace LlTransport {
unsigned records=0;
bool Rdma(const Args*){return true;}
bool SameHost(const Args* a,unsigned peer){return peer/a->host==a->rank/a->host;}
void Records(const Args* a,unsigned peer,uint64_t offset,LocalTensor<uint8_t> src,
             unsigned n,unsigned stride,bool,LocalTensor<uint8_t>) {
    assert(!kCombineControlL2 && SameHost(a,peer));++records;
    GlobalTensor<uint8_t> d{a->peerMems[peer]+offset,false};
    DataCopyPad(d,src,{uint16_t(n),32,0,stride-32,0});
}
}
struct Profile {
    uint64_t DetailClock(){return 0;}
    void Mark(unsigned){} void Elapsed(unsigned,uint64_t){}
};
struct ReadyView {
    Args* args_;uint64_t control_=0;
''' + tensor + r'''
};
template<unsigned Group> struct Publisher {
    Args* args_;uint64_t control_=0;
    unsigned core_=48,notifyBegin_=48,coreNum_=56,recvCores_=TEST_RECV;
    static constexpr unsigned kCombinePrepareRowBytes=72*32;
    Profile profile_;
    float zero[72*8]{},input[72*8]{},work[128]{},sum[8]{},publication[32*8]{};
    LocalTensor<float> prepareZero_{zero},prepareInput_{input},prepareWork_{work},prepareSum_{sum},notifyUb_{publication};
    template<HardEvent E> void Sync(){}
    void WaitFlags(GlobalTensor<float> g,LocalTensor<float>,LocalTensor<float>,LocalTensor<float>,unsigned n,bool) {
        assert(g.cached==kCombineControlL2);
        for(unsigned c=0;c<n;++c)assert(g.p[c*8]==1);
    }
''' + control + notify + r'''
};
template<unsigned Group> void Check() {
    // 32 receive rows extend beyond 32 KiB; the real private-ready region
    // ends at 64 KiB. Check that the test backing store covers every write.
    static_assert(COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET+TEST_RECV*512<=65536);
    std::vector<std::vector<uint64_t>> memory(4*Group,std::vector<uint64_t>(65536/8));
    Args a;a.rank=2*Group-1;a.host=Group;a.peerMems.resize(4*Group);
    for(unsigned p=0;p<4*Group;++p)a.peerMems[p]=reinterpret_cast<uint8_t*>(memory[p].data());
    controls=LlTransport::records=0;
    for(unsigned notifier=0;notifier<8;++notifier) {
        Publisher<Group> k{&a};k.core_+=notifier;
        auto off=COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET+notifier*k.kCombinePrepareRowBytes;
        auto* row=reinterpret_cast<float*>(a.peerMems[a.rank]+off);
        for(unsigned c=0;c<56;++c)row[c*8]=1;
        k.NotifyServerCard();
        for(unsigned c=0;c<56;++c)assert(row[c*8]==0);
    }
    assert(controls==8+Group);
    assert(LlTransport::records==(kCombineControlL2?0:Group));
    for(unsigned peer=0;peer<4*Group;++peer) {
        auto* row=reinterpret_cast<float*>(a.peerMems[peer]+COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET);
        for(unsigned c=0;c<TEST_RECV;++c)for(unsigned member=0;member<Group;++member) {
            const bool written=peer/Group==1 && member==a.rank%Group;
            assert(row[c*128+member*8]==float(written));
        }
    }
    ReadyView r{&a};
    assert(r.Tensor(a.rank,0).cached==kCombineControlL2);
    bool rejected=false;try {r.Tensor(3*Group,0);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}
int main(){Check<2>();Check<4>();Check<16>();}
'''
    cpp, exe = tmp_path / "control.cpp", tmp_path / "control.exe"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}", f"-DTEST_RECV={recv}", "-DDEEPEP_LL_COMBINE_NOTIFY=0",
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=30)
