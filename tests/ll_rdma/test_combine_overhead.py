"""CPU execution of production combine control and immutable-probe metadata."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / 'csrc/kernels/combine/low_latency'


def block(source, marker):
    start = source.index(marker)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def run_cpp(tmp_path, source, name):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for the CPU C++ model')
    cpp = tmp_path / (name + '.cpp')
    exe = tmp_path / (name + '.exe')
    # Isolated control models use a no-op diagnostic sink. The recorder itself
    # is executed with profiling ON/OFF in ll_cpp_bench/test_profile.py.
    cpp.write_text('''#include "kernels/combine/low_latency/combine_profile_layout.h"
struct NoProfile {
 void Mark(CombineProfile::Word) {} void Set(CombineProfile::Word, uint64_t) {}
 uint64_t DetailClock() { return 0; } void Elapsed(CombineProfile::Word, uint64_t) {}
} profile_;
''' + source, encoding='utf-8')
    subprocess.run([zig, 'c++', '-std=c++17', '-O1', '-UNDEBUG',
                    '-I' + str(tmp_path), '-I' + str(ROOT / 'csrc'), str(cpp), '-o', str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_active_prepare_and_ready_ranges(tmp_path):
    source = (KERNEL / 'combine_server_dedup_kernel.h').read_text(encoding='utf-8')
    notify = block(source, '__aicore__ inline void NotifyServerCard()')
    run_cpp(tmp_path, r'''
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
enum class HardEvent { V_S,V_MTE3,S_MTE3 };
template<class T> struct LocalTensor {
 T* p; void SetValue(uint32_t i,T x){p[i]=x;}
 template<class U> LocalTensor<U> ReinterpretCast(){return {reinterpret_cast<U*>(p)};}
};
template<class T> struct GlobalTensor { T* p; };
template<class T> void Duplicate(LocalTensor<T> t,T v,uint32_t n){for(uint32_t i=0;i<n;++i)t.p[i]=v;}
struct CopyParams {uint16_t count;uint32_t bytes,src,dst,reserved;};
template<class T> void DataCopyPad(GlobalTensor<T> g,LocalTensor<T> u,CopyParams c){
 for(uint32_t i=0;i<c.bytes/sizeof(T);++i)g.p[i]=u.p[i];
}
struct Args {uint32_t rank;};
constexpr uint32_t kCombinePrepareRowBytes=72*32;
constexpr uint64_t COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET=0;
constexpr uint64_t COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET=18432;
std::vector<std::vector<uint8_t>> ready;
namespace LlTransport {
void Records(Args*,uint32_t peer,uint64_t offset,LocalTensor<uint8_t> u,uint32_t count,uint32_t stride,bool,LocalTensor<uint8_t>){
 for(uint32_t r=0;r<count;++r)for(uint32_t b=0;b<32;++b)
  ready[peer][offset-COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET+r*stride+b]=u.p[r*32+b];
}}
template<uint32_t Group> struct Test {
 uint32_t coreNum_,recvCores_,core_,notifyBegin_; Args a{0}; Args* args_=&a; uint64_t control_=0;
 std::vector<float> done,zero,notification,input,work,sum;
 LocalTensor<float> prepareZero_,notifyUb_,prepareInput_,prepareWork_,prepareSum_;
 Test(uint32_t n,uint32_t recv):coreNum_(n),recvCores_(recv),notifyBegin_(n-8),done(8*72*8,13),zero(72*8),notification(36*8),input(72*8),work(128),sum(8){
  prepareZero_={zero.data()};notifyUb_={notification.data()};prepareInput_={input.data()};prepareWork_={work.data()};prepareSum_={sum.data()};
  for(uint32_t row=0;row<8;++row)for(uint32_t c=0;c<n;++c)done[row*72*8+c*8]=1;
 }
 template<HardEvent>void Sync(){}
 GlobalTensor<float> Control(uint32_t,uint64_t off){return {done.data()+off/4};}
 void WaitFlags(GlobalTensor<float> gm,LocalTensor<float>,LocalTensor<float>,LocalTensor<float>,uint32_t n,bool incoming){assert(!incoming);for(uint32_t i=0;i<n;++i)assert(gm.p[i*8]==1);}
''' + notify + r'''
 void Run(){
  ready.assign(Group,std::vector<uint8_t>(36*512,0x5b));
  for(core_=notifyBegin_;core_<coreNum_;++core_)NotifyServerCard();
  for(uint32_t row=0;row<8;++row)for(uint32_t x=0;x<72*8;++x)
   assert(done[row*72*8+x]==(x<coreNum_*8?0:13));
  for(auto& peer:ready)for(uint32_t r=0;r<36;++r)for(uint32_t b=0;b<512;++b){
   if(r<recvCores_ && b<32){float one=1;uint8_t expected=b<4?reinterpret_cast<uint8_t*>(&one)[b]:0;assert(peer[r*512+b]==expected);}
   else assert(peer[r*512+b]==0x5b);
  }
 }
};
int main(){for(uint32_t n=16;n<=72;n+=2)for(uint32_t recv=8;recv<=n/2;++recv){Test<4>(n,recv).Run();Test<16>(n,recv).Run();}}
''', 'combine-control')


def test_mask_validation_and_popcount_once_per_token(tmp_path):
    source = (KERNEL / 'combine_server_dedup_source_aggregate.h').read_text(encoding='utf-8')
    init = block(source, 'for (uint32_t p = 0; p < count_; ++p) {')
    process = block(source, '__aicore__ inline uint32_t Process()')
    run_cpp(tmp_path, r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define __aicore__
struct half{};struct bfloat16_t{};
enum class HardEvent {V_MTE2};
template<HardEvent>void SetFlag(uint32_t){} template<HardEvent>void WaitFlag(uint32_t){}
void Trap(){throw std::runtime_error("invalid mask");}
uint32_t popcounts=0;
template<int>uint32_t GetBitCount(uint64_t x){++popcounts;uint32_t n=0;while(x){x&=x-1;++n;}return n;}
namespace DispatchDedup {constexpr int DISPATCH_DEDUP_DATA_TYPE_FP16=0;}
namespace LlTransport {struct PollDeadline{PollDeadline(bool){}void Check(){}};}
struct Tensor{std::vector<uint32_t> v;uint32_t GetValue(uint32_t i){return v.at(i);}void SetValue(uint32_t i,uint32_t x){v.at(i)=x;}};
struct Test {
 bool enableDedup_=true,uncached_=false;uint32_t count_=3,begin_=17,k_=8,blocks_=15,inputLanes_=2,outputLanes_=1,retiredOutputs_=0,dtype_=0;
 Tensor masks_{{1,0x81,0xff}},pending_{{0,0,0}};
 uint32_t probes=0,clears=0,completed=0;
 void Init(){
''' + init + r'''
 }
 bool Probe(uint32_t token,uint32_t expected){++probes;assert(probes<100);assert(expected==(enableDedup_?uint32_t(__builtin_popcount(masks_.GetValue(token-begin_))):k_)*blocks_);return probes%4==0;}
 void ClearFlags(){++clears;}
 template<class T>void Aggregate(uint32_t token,uint32_t mask,uint32_t ordinal){assert(mask==(enableDedup_?masks_.GetValue(token-begin_):0));assert(ordinal==completed++);}
''' + process + r'''
};
int main(){
 Test t;t.Init();assert(popcounts==3);assert((t.pending_.v==std::vector<uint32_t>{17U|(15U<<16),18U|(30U<<16),19U|(120U<<16)}));t.Process();assert(popcounts==3&&t.completed==3&&t.clears==1&&t.probes==12);
 Test wide;wide.k_=32;wide.blocks_=86;wide.begin_=65533;wide.masks_.v={0xffffffffU,0x80000000U,1};wide.Init();assert((wide.pending_.v[0]>>16)==2752 && (wide.pending_.v[2]&0xffff)==65535);wide.Process();
 for(uint32_t bad:{0U,256U}){Test invalid;invalid.masks_.v[1]=bad;bool trapped=false;try{invalid.Init();}catch(std::runtime_error&){trapped=true;}assert(trapped);}
 Test direct;direct.enableDedup_=false;uint32_t before=popcounts;direct.Init();direct.Process();assert(popcounts==before);
}
''', 'combine-mask')


def test_flag_initializer_preserves_payload_and_bounds(tmp_path):
    (tmp_path / 'kernel_operator.h').write_text(r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
namespace AscendC {
enum class CacheMode {CACHE_MODE_DISABLE};
template<class T>struct LocalTensor {T* p;};
template<class T>struct GlobalTensor {
 uintptr_t p;void SetGlobalBuffer(T* q){p=reinterpret_cast<uintptr_t>(q);}
 void SetL2CacheHint(CacheMode){} GlobalTensor operator[](uint64_t i){return {p+i*sizeof(T)};}
};
struct Params {uint16_t count;uint32_t bytes,src,dst,reserved;};
inline std::vector<uint8_t> banks[2];inline std::vector<uint32_t> writes[2];
inline void DataCopyPad(GlobalTensor<uint8_t> gm,LocalTensor<uint8_t> ub,Params c){
 assert(c.count>0 && c.count<=256 && c.bytes==32 && c.src==0 && c.dst==480);
 const uint64_t address=gm.p-0x1000000000ULL;const uint32_t bank=uint32_t(address/(4ULL<<30));
 const uint64_t offset=address%(4ULL<<30)-(2ULL<<20);assert(bank<2);
 for(uint32_t i=0;i<c.count;++i){const uint64_t dst=offset+i*512;assert(dst%512==480 && dst+32<=banks[bank].size());
  for(uint32_t b=0;b<32;++b){assert(ub.p[i*32+b]==0);banks[bank][dst+b]=0;}++writes[bank][dst/512];}
}
}
''', encoding='utf-8')
    run_cpp(tmp_path, r'''
#include "kernels/combine/low_latency/combine_clear_flags.h"
int main(){
 uint8_t zeros[CombineDedup::kClearFlagUbBytes]{};
 for(uint32_t cores=16;cores<=72;cores+=2)for(uint64_t blocks:{0ULL,1ULL,15ULL,960ULL,8193ULL}){
  for(uint32_t bank=0;bank<2;++bank){AscendC::banks[bank].assign(blocks*512,0x69);AscendC::writes[bank].assign(blocks,0);}
  for(uint32_t core=0;core<cores;++core)CombineDedup::ClearResultFlags(reinterpret_cast<uint8_t*>(0x1000000000ULL),blocks,{zeros},core,cores);
  for(uint32_t bank=0;bank<2;++bank)for(uint64_t b=0;b<blocks;++b){assert(AscendC::writes[bank][b]==1);
   for(uint32_t i=0;i<512;++i)assert(AscendC::banks[bank][b*512+i]==(i<480?0x69:0));}
 }
}
''', 'combine-flag-clear')
