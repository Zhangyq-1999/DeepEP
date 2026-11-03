from pathlib import Path
import os
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_fullmesh_first_hop_conserves_routes_and_minimizes_clos(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the native policy test")
    exe = tmp_path / "fullmesh-first-hop.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    "-I" + str(ROOT / "csrc"),
                    str(Path(__file__).with_name("fullmesh_first_hop_test.cpp")),
                    "-o", str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    assert "PASS fullmesh first-hop" in result.stdout
    print(result.stdout)


def test_sender_scopes_policy_and_keeps_earlier_members():
    code = (ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h").read_text(encoding="utf-8")
    assert "Group == kLlPhysicalRanksPerHost && LlTransport::Rdma(args)" in code
    assert "LlFirstHopForMask(routeIds_" in code
    assert "meshFirstK" not in code
    assert "LlRelayMemberMask(mask, firstK)" in code
    assert "firstK + 1U" not in code
    assert "for (uint32_t k = 0U; members != 0U && k < k_; ++k)" in code


@pytest.mark.parametrize("ranks_per_host", [16, 4])
def test_actual_classify_and_per_host_selection(tmp_path, ranks_per_host):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for the native policy test')
    code = (ROOT / 'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h').read_text(encoding='utf-8')
    start = code.index('{', code.index('void Classify('))
    depth = 1
    for end in range(start + 1, len(code)):
        depth += (code[end] == '{') - (code[end] == '}')
        if depth == 0:
            break
    body = code[start + 1:end]
    source = tmp_path / 'multi-first-hop.cpp'
    source.write_text(r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>
#include "kernels/dispatch/low_latency/common/ll_fullmesh_first_hop.h"
using namespace DispatchDedup;
constexpr unsigned Group=16;
static unsigned reads=0;
template<class T> struct Tensor {
    std::vector<T>& data;bool ids=false;
    T GetValue(unsigned i) {if(ids)++reads;return data.at(i);}
    void SetValue(unsigned i,T value) {data.at(i)=value;}
};
struct Args {unsigned rank,rankSize;};
struct Fixture {
    Args* args_;
    unsigned start_=1,k_,localExperts_=2;
    Tensor<int32_t> routeIds_;
    Tensor<uint32_t> serverState_,firstHopMasks_;
    void Classify(uint32_t token,uint32_t& directMask,uint32_t& groups) {
''' + body + r'''
    }
};
static void Check(unsigned world,unsigned rank,const std::vector<int32_t>& input) {
    const unsigned k=input.size(),base=2*k;
    std::vector<int32_t> ids(3*k,-1);
    std::copy(input.begin(),input.end(),ids.begin()+base);
    std::vector<uint32_t> states(world/16),masks(16,0xdeadbeef);
    Args args{rank,world};Fixture f{&args,1,k,2,{ids,true},{states},{masks}};
    unsigned direct=0,groups=0;reads=0;f.Classify(1,direct,groups);
    assert(reads==k);
    std::map<unsigned,unsigned> byHost;
    for(unsigned i=0;i<k;++i)byHost[unsigned(input[i])/2/16]|=1U<<i;
    unsigned expectedDirect=0;std::set<unsigned> expectedGroups,actualGroups;
    for(auto [host,mask]:byHost) {
      if(host==rank/16 || !(mask&(mask-1))) expectedDirect|=mask;
      else expectedGroups.insert(mask);
    }
    assert(direct==expectedDirect && groups==expectedGroups.size());
    std::array<unsigned,32> visits{};
    for(unsigned i=0;i<k;++i)if(direct&(1U<<i))++visits[i];
    for(unsigned group=0;group<groups;++group) {
      const unsigned mask=masks[group];assert(actualGroups.insert(mask).second);
      const unsigned selected=LlFirstHopForMask(Tensor<int32_t>{ids,true},base,k,2,mask,true);
      assert(selected<k && (mask&(1U<<selected)));
      const unsigned host=unsigned(input[selected])/2/16;
      assert(host!=rank/16);
      std::array<unsigned,4> counts{};
      for(unsigned i=0;i<k;++i)if(mask&(1U<<i)) {
        assert(unsigned(input[i])/2/16==host);++counts[(unsigned(input[i])/2%16)/4];
      }
      unsigned expected=32,best=0,earliest=32;
      for(unsigned i=0;i<k;++i)if(mask&(1U<<i)) {
        earliest=std::min(earliest,i);
        const auto n=counts[(unsigned(input[i])/2%16)/4];
        if(n>best){best=n;expected=i;}
      }
      assert(selected==expected);
      assert(LlFirstHopForMask(Tensor<int32_t>{ids},base,k,2,mask,false)==earliest);
      const unsigned members=LlRelayMemberMask(mask,selected);
      ++visits[selected];
      for(unsigned i=0;i<k;++i)if(members&(1U<<i))++visits[i];
    }
    assert(actualGroups==expectedGroups && reads<=2*k);
    for(unsigned i=0;i<k;++i)assert(visits[i]==1);
}
int main() {
    // A singleton on host1 must not become host2's first hop.
    Check(64,0,{16*2,32*2,33*2});
    // Two independent host majorities require distinct selected top-k slots.
    Check(64,0,{16*2,20*2,21*2,32*2,36*2,37*2});
    std::mt19937 rng(28029);unsigned cases=2;
    for(unsigned world: {32U,64U,128U,256U}) for(unsigned rank: {0U,world/2,world-1})
      for(unsigned k=1;k<=32;++k) for(unsigned sample=0;sample<30;++sample) {
        std::set<int32_t> unique;std::vector<int32_t> ids;
        while(ids.size()<k) {int32_t e=rng()%(world*2);if(unique.insert(e).second)ids.push_back(e);}
        Check(world,rank,ids);++cases;
      }
    std::cout<<"PASS actual multi-host Classify/first-hop: "<<cases<<" cases; independent masks, no lost/duplicate routes, <=2K route reads\n";
}
''', encoding='utf-8')
    if ranks_per_host == 4:
        text = source.read_text(encoding='utf-8')
        text = text.replace('constexpr unsigned Group=16;', 'constexpr unsigned Group=4;')
        text = text.replace('/16', '/Group').replace('%16', '%Group')
        text = text.replace('Check(64,0,{16*2,32*2,33*2});', 'Check(8,0,{2,8,10});')
        text = text.replace('Check(64,0,{16*2,20*2,21*2,32*2,36*2,37*2});', 'Check(8,4,{0,2,4,8,10,12});')
        text = text.replace('{32U,64U,128U,256U}', '{8U}').replace('k<=32', 'k<=16')
        source.write_text(text, encoding='utf-8')
    exe = tmp_path / 'multi-first-hop.exe'
    subprocess.run([zig, 'c++', '-std=c++17', '-O2', '-UNDEBUG', '-I' + str(ROOT / 'csrc'),
                    f'-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={ranks_per_host}',
                    str(source), '-o', str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    assert 'PASS actual multi-host Classify/first-hop' in result.stdout
    print(result.stdout)
