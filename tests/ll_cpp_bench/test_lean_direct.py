"""Execute production DIRECT traversal/packing, including the UB route cache."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("mode", [20, 24, 25, 26, 27, 28, 29, 30])
def test_direct_snapshot_lanes_and_fallback(tmp_path, mode, physical):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    kernel = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h").read_text(encoding="utf-8")
    methods = kernel[kernel.index("    __aicore__ inline bool PreloadDirectRoutes()"):
                     kernel.index("    // M4→M5")]
    # Replace transport only; execute the actual Vector selection, ownership,
    # source route reads, input load, flag initialization, packing and events.
    source = r'''
#define main ReadyProtocolChecks
#include "ready_test.cpp"
#undef main
#include "kernels/combine/low_latency/combine_return_schedule.h"
#include "kernels/combine/low_latency/combine_server_dedup_select.h"
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
namespace AscendC {
struct CopyArgs {uint32_t db,sb,dr,sr;};
template<class T> void Copy(LocalTensor<T> d,LocalTensor<T> s,uint64_t n,uint8_t reps,CopyArgs a) {
    for(unsigned r=0;r<reps;++r)for(unsigned i=0;i<n;++i)
        d.p[r*a.dr*32/sizeof(T)+i]=s.p[r*a.sr*32/sizeof(T)+i];
}
}
struct Batch {
    struct Packet {unsigned peer,token,slot;std::vector<uint8_t> bytes;};
    std::vector<Packet> packets;
    void Init(const DispatchDedup::CommArgs*,unsigned){}
    void SubmitPending(){}
};
void SendReturnPacket(Batch& b,const DispatchDedup::CommArgs*,unsigned peer,uint64_t off,
                      LocalTensor<uint8_t> src,unsigned bytes,bool defer) {
    assert(!defer);off-=COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET;
    unsigned index=off/bytes;
    b.packets.push_back({peer,index/8,index%8,{src.p,src.p+bytes}});
}
template<unsigned Group> struct Direct {
    DispatchDedup::CommArgs a{},*args_=&a;
    bool enableDedup_=true,directRoutesCached_=false;
    unsigned tokens_=0,core_=0,sendCores_=28,lanes_=8,blocks_=15,h_=3584,topK_=8;
    unsigned laneBytes_=7200,resultStride_=7680;
    uint64_t selected_=0,window_=0;
    CombineProfile::View profile_;
    Batch returnBatch_;
    std::vector<int32_t> types,table,typ,idx,packed,cache;
    std::vector<uint8_t> mask,ready,payload,result;
    std::vector<uint16_t> expert;
    GlobalTensor<int32_t> types_,table_;
    GlobalTensor<uint16_t> expert_;
    LocalTensor<int32_t> typeUb_,indexUb_,packedUb_,tableUb_;
    LocalTensor<uint8_t> maskUb_,resultUb_;
    LocalTensor<float> readyWork_;
    LocalTensor<uint16_t> payloadUb_;
    template<HardEvent E>void Sync(){SetFlag<E>(7);WaitFlag<E>(7);}
    Direct(unsigned rows,unsigned world,unsigned rank,unsigned sender,bool dedup,unsigned h):tokens_(rows),core_(sender),enableDedup_(dedup),h_(h) {
        a.rankSize=world;a.rank=rank;a.reserved0=Group;a.reservedConfig[0]=LlRdma::kBackend;
        blocks_=(h*2+479)/480;laneBytes_=blocks_*480;resultStride_=blocks_*512;
        types.resize(rows);table.resize(rows*3);expert.resize(rows*h);
        for(unsigned row=0;row<rows;++row) {
            types[row]=row%5==0?2:1;
            table[row*3]=types[row]==1 || !dedup ? (row*13+row/7)%world : world+100;
            table[row*3+1]=row;table[row*3+2]=row%8;
            for(unsigned i=0;i<h;++i)expert[row*h+i]=uint16_t(row*73+i);
        }
        unsigned padded=EpCombineAlign(std::max(rows,1U),64U);
        typ.resize(padded);idx.resize(padded);packed.resize(padded);
        cache.resize(kEpServerDedupCombineTableRows*3+8,0x11223344);
        mask.resize(EpCombineAlign((rows+7)/8+32,256));ready.resize(256+32,0xa5);
        payload.resize(lanes_*laneBytes_);result.resize(lanes_*resultStride_+32,0x5b);
        typeUb_={typ.data()};indexUb_={idx.data()};packedUb_={packed.data()};tableUb_={cache.data()};
        maskUb_={mask.data()};readyWork_={reinterpret_cast<float*>(ready.data())};
        payloadUb_={reinterpret_cast<uint16_t*>(payload.data())};resultUb_={result.data()};
        types_.SetGlobalBuffer(types.data());table_.SetGlobalBuffer(table.data());expert_.SetGlobalBuffer(expert.data());
    }
''' + methods + r'''
    void Check() {
        std::vector<unsigned> expected;
        unsigned owner=0;
        for(unsigned step=0;step<a.rankSize/Group;++step) {
            unsigned server=CombineReturnServer<Group>(a.rankSize/Group,a.rank/Group,step);
            for(unsigned row=0;row<tokens_;++row)if((!enableDedup_ || types[row]==1) && table[row*3]/Group==server) {
                if(owner++%sendCores_==core_)expected.push_back(row);
            }
        }
        if constexpr (kCombineDirectRemoteFirst && Group == DispatchDedup::kLlPhysicalRanksPerHost) {
            std::stable_partition(expected.begin(), expected.end(), [&](unsigned row) {
                return unsigned(table[row*3])/Group != a.rank/Group;
            });
        }
        ReturnDirectTokens();
        assert(returnBatch_.packets.size()==expected.size());
        for(unsigned i=0;i<expected.size();++i) {
            unsigned row=expected[i];auto& p=returnBatch_.packets[i];
            assert(p.token==row && p.peer==unsigned(table[row*3]) && p.slot==row%8);
            for(unsigned x=0;x<h_;++x) {
                uint16_t value;std::memcpy(&value,p.bytes.data()+x*2/480*512+x*2%480,2);
                assert(value==expert[row*h_+x]);
            }
            for(unsigned block=0;block<blocks_;++block)for(unsigned j=0;j<8;++j) {
                float f;std::memcpy(&f,p.bytes.data()+block*512+480+j*4,4);assert(f==1.f);
            }
        }
        if(owner)assert(directRoutesCached_==(kCombineDirectRouteCache && tokens_<=512));
        if(directRoutesCached_)assert(std::equal(table.begin(),table.end(),cache.begin()));
        for(unsigned i=512*3;i<cache.size();++i)assert(cache[i]==0x11223344);
        for(unsigned i=256;i<ready.size();++i)assert(ready[i]==0xa5);
        for(unsigned i=lanes_*resultStride_;i<result.size();++i)assert(result[i]==0x5b);
        NoLiveEvents();
        assert(deferredMte2.empty());
    }
};
int main() {
    for(unsigned hosts:{1U,2U,4U})for(unsigned rank:{0U,hosts*16-1U})for(bool dedup:{false,true})
    for(unsigned rows:{0U,1U,17U,256U,511U,512U,513U,2049U})for(unsigned c=0;c<28;++c)
        Direct<16>(rows,hosts*16,rank,c,dedup,256).Check();
    for(unsigned h:{256U,3584U,7168U})for(unsigned rows:{32U,512U,513U})for(unsigned c:{0U,1U,27U})
        Direct<4>(rows,8,4,c,true,h).Check();
    // Empty selections must retire speculative DMA before cache/arena reuse.
    for(unsigned rows:{17U,512U,513U}) {
        Direct<16> empty(rows,64,0,0,true,256);
        std::fill(empty.types.begin(),empty.types.end(),2);empty.Check();
    }
    // DIRECT-only rank validation still rejects invalid selected rows.
    for(int bad:{-1,64}) {
        Direct<16> invalid(17,64,0,0,true,256);invalid.table[3]=bad;
        bool trapped=false;
        try {invalid.PrepareDirectRows();}catch(const std::runtime_error&){trapped=true;}
        assert(trapped);NoLiveEvents();assert(deferredMte2.empty());
    }
}
'''
    cpp, exe = tmp_path / "direct.cpp", tmp_path / "direct.exe"
    cpp.write_text(source, encoding="utf-8")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_TEST_DEFERRED_MTE2=1",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}", f"-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={physical}",
                    "-I" + str(ROOT / "tests/ll_cpp_bench"),
                    "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"), "-I" + str(ROOT / "csrc"),
                    str(cpp), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=60)
