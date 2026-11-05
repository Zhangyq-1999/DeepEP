"""Run the production M4 Send with delayed DMA and delayed NIC consumption."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("profile", [False, True])
def test_stream_packet_lifetime_and_eager_head(tmp_path, physical, profile):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the asynchronous transport model")
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    start = text.index("    __aicore__ inline void Send(uint32_t lane,")
    end = text.index("\n#if DEEPEP_LL_COMBINE_READY == 1 ||", start)
    source = r'''
#include <cassert>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>
#define DEEPEP_LL_RDMA 1
#define DEEPEP_TEST_DEFERRED_MTE3 1
#include "kernels/combine/low_latency/combine_profile_device.h"
#include "kernels/combine/low_latency/combine_rdma_return.h"
#include "kernels/combine/low_latency/ep_memory_server_dedup_buffer_layout.h"
using namespace AscendC;
using namespace CombineDedup;
unsigned packed=0;
namespace AscendC {
struct CopyParams {uint32_t dstStride,srcStride,dstRepeat,srcRepeat;};
inline void Copy(LocalTensor<int32_t> dst,LocalTensor<int32_t> src,uint64_t n,uint8_t repeats,CopyParams p) {
    for(unsigned b=0;b<repeats;++b)
        std::copy_n(src.p+b*p.srcRepeat*8U,n,dst.p+b*p.dstRepeat*8U);
    if(n==56U)++packed;
}
}
struct Sender {
    ReturnBatch* rdmaBatch_;
    DispatchDedup::CommArgs* args_;
    CombineProfile::View profile_;
    LocalTensor<uint8_t> outputs_;
    LocalTensor<uint16_t> low_;
    uint32_t blocks_,packetBytes_;
    uint64_t returnOffset_;
''' + text[start:end] + r'''
};
int main() {
    cores=56;core=31;
    const unsigned host=DispatchDedup::kLlPhysicalRanksPerHost;
    unsigned cases=0;
    for(unsigned world:{2U*host,4U*host}) for(unsigned rank:{0U,world-1U})
    for(unsigned blocks:{1U,15U,128U}) for(unsigned lanes:{1U,2U,8U})
    for(unsigned pattern:{0U,1U,2U}) for(unsigned batchSize:{1U,2U,3U,7U}) {
        const unsigned bytes=blocks*512U,capacity=LlRdma::kCoreBytes/bytes,n=capacity+5U;
        const uint64_t stride=bytes+world*1024U;
        std::vector<uint8_t> local((n+1U)*stride),remote(local.size()),stage(cores*LlRdma::kCoreBytes),scratch(256U);
        std::vector<uint8_t> ub(lanes*bytes),low(blocks*480U);
        std::vector<std::vector<uint8_t>> expected;
        std::vector<unsigned> targets,remoteTargets,postOrder;
        Model::local=local.data();Model::remote=remote.data();Model::pending.clear();
        Model::onQuiet={};Model::puts=Model::quiets=0;deferredMte3.clear();issuedMte3=0;packed=0;
        DispatchDedup::CommArgs a{};a.rank=rank;a.rankSize=world;a.reserved0=host;a.peerMemBytes=local.size();
        a.reservedAddresses[0]=local.data();a.reservedAddresses[1]=stage.data();
        a.reservedConfig[0]=LlRdma::kBackend;a.reservedConfig[1]=cores;
        for(unsigned peer=rank/host*host;peer<(rank/host+1U)*host;++peer)a.peerMems[peer]=local.data()+peer*1024U;
        ReturnBatch batch;batch.Init(&a,bytes);
        Sender sender{&batch,&a,{}, {ub.data()}, {reinterpret_cast<uint16_t*>(low.data())},blocks,bytes,0U};
#ifdef DEEPEP_LL_COMBINE_PROFILE
        std::vector<uint64_t> record(CombineProfile::kWords*(core+1U));
        CombineProfile::DeviceProfile profile;profile.Init(reinterpret_cast<uint8_t*>(record.data()),2);
        batch.SetCombineProfile(&profile);sender.profile_.Bind(&profile);
#endif
        const auto check=[&](const Model::Request& r) {
            const unsigned i=unsigned((r.dst-remote.data()-r.peer*1024U)/stride);
            assert(r.qp==core && r.peer==targets.at(i) && r.bytes==bytes);
            assert(std::memcmp(r.src,expected.at(i).data(),bytes)==0);
        };
        Model::before=check; // Delay NIC consumption until Quiet, even across UB reuse.
        unsigned current=0,packedPosts=0;
        Model::onPost=[&](Model::Request& r) {
            check(r);
            unsigned i=unsigned((r.dst-remote.data()-r.peer*1024U)/stride);
            assert(packed==current+1U); // Actual Send has queued both current Vector copies.
            if(i!=current){assert(i+1U==current);++packedPosts;}
            postOrder.push_back(i);
        };
        for(unsigned lane=0;lane<lanes;++lane) {
            SetFlag<HardEvent::MTE3_V>(lane);
            for(unsigned b=0;b<blocks;++b)std::memset(ub.data()+lane*bytes+b*512U+480U,0x3f,32U);
        }
        for(unsigned i=0;i<n;++i) {
            current=i;
            const unsigned index=i%batchSize,count=std::min(batchSize,n-i+index);
            const bool cross=pattern==0 || (pattern==2 && i%3U!=1U);
            const unsigned other=((rank/host+1U+i%(world/host-1U))%(world/host))*host;
            const unsigned peer=cross?other+i%host:rank/host*host+i%host;
            targets.push_back(peer);if(cross)remoteTargets.push_back(peer);
            for(unsigned j=0;j<low.size();++j)low[j]=uint8_t((i*13U+j)%251U);
            expected.emplace_back(bytes,0x3f);
            for(unsigned b=0;b<blocks;++b)std::memcpy(expected.back().data()+b*512U,low.data()+b*480U,480U);
            const bool defer=CombineStreamDefer(index,count,lanes);
            sender.returnOffset_=i*stride;
            sender.Send(i%lanes,peer,i,0U,defer);
            if(index==0U || index+1U==count || lanes==1U)assert(!batch.HasStaged());
            if(cross && index==0U)assert(postOrder.back()==i); // Never delay the first packet.
        }
        assert(!batch.HasStaged()); // Before M5, every packet has a submitted WQE.
        for(unsigned lane=0;lane<lanes;++lane)WaitFlag<HardEvent::MTE3_V>(lane);
        std::fill(ub.begin(),ub.end(),0xee); // M5 aliases this entire payload arena.
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
        if(lanes==1U || batchSize<=2U)assert(packedPosts==0U);
        for(unsigned i=0;i<n;++i) {
            const auto* base=LlTransport::RemoteWrite(&a,targets[i])?remote.data():local.data();
            assert(std::memcmp(base+targets[i]*1024U+i*stride,expected[i].data(),bytes)==0);
        }
#ifdef DEEPEP_LL_COMBINE_PROFILE
        profile.Finish(rank,world,n);
        const auto* r=record.data()+core*CombineProfile::kWords;
        assert(r[CombineProfile::RdmaStageWaitCalls]==Model::puts);
        assert(r[CombineProfile::PartialPackedSubmitCalls]==packedPosts);
#endif
        for(auto& direction:events)for(bool live:direction)assert(!live);
        ++cases;
    }
    std::cout<<"PASS "<<cases<<" stream staging cases\n";
}
'''
    cpp, exe = tmp_path / "stream.cpp", tmp_path / "stream.exe"
    cpp.write_text(source, encoding="utf-8")
    (tmp_path / "shmem.h").write_text("#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    "-DDEEPEP_LL_COMBINE_READY=7", f"-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={physical}",
                    *(["-DDEEPEP_LL_COMBINE_PROFILE=1"] if profile else []),
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tests/ll_rdma/stubs"),
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=120)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=90)
    assert "PASS 432 stream staging cases" in result.stdout
