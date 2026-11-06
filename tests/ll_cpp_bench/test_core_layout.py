"""Check physical placement, task coverage and ready-row ownership without NPU."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", [12, 18, 19])
@pytest.mark.parametrize("recv", [0, 32])
def test_physical_placement_and_logical_ownership(tmp_path, mode, recv):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU checks")
    kernel = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h").read_text(encoding="utf-8")
    # Execute the real Init mapping, including unsupported-layout rejection.
    begin = kernel.index("        coreNum_ = GetBlockNum();")
    end = kernel.index("        args_ =", begin)
    cpp = tmp_path / "layout.cpp"
    cpp.write_text(r'''
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
#include "producer_map.hpp"
using namespace CombineDedup;
unsigned physical, active;
unsigned GetBlockIdx() {return physical;}
unsigned GetBlockNum() {return active;}
void Trap() {throw std::runtime_error("unsupported split");}
struct Kernel {
    unsigned core_,coreNum_,recvCores_,sendCores_,notifyBegin_;
    void Init() {
''' + kernel[begin:end] + r'''
    }
};
int main() {
    for (active=16;active<=72;active+=2) {
        const unsigned receivers=EpCombineRecvCoreCount(active);
        if (!receivers) {
            physical=0;bool rejected=false;
            try { Kernel k;k.Init(); } catch(const std::runtime_error&) {rejected=true;}
            assert(rejected);continue;
        }
        const unsigned senders=active-receivers;
        std::vector<unsigned> ids(active),receiveIds(receivers),notifyIds(8);
        // M2 prepare publication and later flag readers retain logical rows.
        for (physical=0;physical<active;++physical) {
            Kernel k;k.Init();
            assert(k.core_<active);++ids[k.core_];
            assert(CombinePhysicalCore(k.core_,active)==physical);
            const bool receive=CombineIsReceiveCore(physical,active,receivers);
            assert(receive==(k.core_>=k.sendCores_));
            if constexpr(kCombineInterleavedCores) assert(receive==(physical%2==0));
            else assert(k.core_==physical);
            if(receive) {
                const auto local=CombineReceiveCoreIndex(physical,active,receivers);
                assert(local<receivers && local==k.core_-k.sendCores_);++receiveIds[local];
            }
            if(k.core_>=k.notifyBegin_) {
                assert(receive);++notifyIds[k.core_-k.notifyBegin_];
                if constexpr(kCombineInterleavedCores)
                    assert(physical==2U*(receivers-8U+k.core_-k.notifyBegin_));
            }
        }
        for(auto n:ids)assert(n==1);
        for(auto n:receiveIds)assert(n==1);
        for(auto n:notifyIds)assert(n==1);
        // Tail diagnostics must identify the actual AIV/QP owner, not its
        // logical ready-row index. Exercise every sender/receiver owner.
        std::vector<int32_t> types(2*active),table(6*active);
        for(unsigned kind=0;kind<2;++kind)for(unsigned i=0;i<active;++i) {
            unsigned row=kind*active+i;
            types[row]=kind+1;table[3*row]=kind;table[3*row+1]=i;
        }
        auto producers=llbench::return_producers(types,table,0,4,active,1,4,4,active,receivers,false);
        for(unsigned kind=0;kind<2;++kind)for(unsigned i=0;i<active;++i) {
            unsigned logical=kind?senders+i%receivers:i%senders;
            unsigned expected=kCombineInterleavedCores?(kind?2*(i%receivers):2*(i%senders)+1):logical;
            assert(((producers[kind*active+i]>>16U)&255U)==expected);
        }
        // Empty/tiny/large task sets: M2, DIRECT, partial/final work must each
        // be covered exactly once after moving the logical owners to AIVs.
        for(unsigned n:{0U,1U,7U,32U,48U,64U,72U,256U,2048U}) {
            std::vector<unsigned> prepare(n),direct(n),partial(n);
            for(physical=0;physical<active;++physical) {
                Kernel k;k.Init();
                const unsigned begin=k.core_*(n/active)+std::min(k.core_,n%active);
                const unsigned count=n/active+(k.core_<n%active);
                for(unsigned i=begin;i<begin+count;++i)++prepare[i];
                if(k.core_<senders)for(unsigned i=k.core_;i<n;i+=senders)++direct[i];
                else for(unsigned i=CombineReceiveCoreIndex(physical,active,receivers);i<n;i+=receivers)++partial[i];
            }
            for(auto* rows:{&prepare,&direct,&partial})for(auto count:*rows)assert(count==1);
        }
        // Notification covers every local peer once, including G16's second
        // round and all four host groups in EP64. No pair-of-servers shortcut.
        for(unsigned group:{2U,4U,16U})for(unsigned servers:{1U,2U,4U})for(unsigned host=0;host<servers;++host) {
            std::vector<unsigned> peers(group*servers);
            for(physical=0;physical<active;++physical) {
                Kernel k;k.Init();
                if(k.core_<k.notifyBegin_)continue;
                for(unsigned local=k.core_-k.notifyBegin_;local<group;local+=8U)++peers[host*group+local];
            }
            for(unsigned peer=0;peer<peers.size();++peer)assert(peers[peer]==unsigned(peer/group==host));
        }
    }
    if constexpr(kCombineInterleavedCores)assert(EpCombineRecvCoreCount(55)==0);
}
''', encoding="utf-8")
    exe = tmp_path / "layout.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}", f"-DDEEPEP_LL_COMBINE_RECV_CORES={recv}",
                    "-I" + str(ROOT / "csrc"), "-I" + str(ROOT / "tests/ll_cpp_bench"),
                    str(cpp), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=30)
