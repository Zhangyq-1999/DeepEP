"""Run production transfer adapters with delayed MTE/NIC and SDK contract stubs."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("mode", [9, 10, 11, 12, 13, 14, 15, 16, 17])
@pytest.mark.parametrize("profile", [False, True])
def test_transfer_lifetimes(tmp_path, physical, mode, profile):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for asynchronous transport checks")
    (tmp_path / "shmem.h").write_text("#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    exe = tmp_path / "transfer.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}", f"-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={physical}",
                    *(["-DDEEPEP_LL_COMBINE_PROFILE=1"] if profile else []),
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tests/ll_rdma/stubs"),
                    "-I" + str(ROOT / "csrc"), str(Path(__file__).with_name("transfer_batch_test.cpp")),
                    "-o", str(exe)], check=True, timeout=120)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=60)
    assert "PASS 40 transfer cases" in result.stdout


def test_private_wqe_ub_budget(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for UB layout checks")
    source = tmp_path / "layout.cpp"
    source.write_text(r'''
#include <cassert>
#include <iostream>
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
using namespace CombineDedup;
int main(){
    for(unsigned tokens:{0U,32U,256U,2048U,10000U})for(unsigned bs:{32U,48U,64U,72U,256U}){
        auto p=EpCombineM45Layout(tokens,bs,3584,8,28);
        assert(p.inputLanes && p.outputLanes && p.totalBytes<=kEpServerDedupCombineRecvUbBytes);
        const unsigned metadata=EpCombineM4TaskBytes(tokens,28)+3*EpCombineReadyTaskBytes(tokens,28)
            +EpCombineAlign((EpCombineReadyTaskBytes(tokens,28)/4U+7U)/8U,32U)+384+32*8*8;
        const unsigned end=p.phaseOffset+metadata+15*480+((DEEPEP_LL_COMBINE_READY==9 || DEEPEP_LL_COMBINE_READY>=11)?128:0);
        assert(end<=p.inputOffset);
        auto final=EpCombineM5Layout(p,100);
        assert(final.inputLanes);
        std::cout<<p.inputLanes<<','<<p.outputLanes<<','<<final.inputLanes<<'\n';
    }
}
''')
    layouts = []
    for mode in (4, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        exe = tmp_path / f"layout-{mode}.exe"
        subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                        f"-DDEEPEP_LL_COMBINE_READY={mode}", "-I" + str(ROOT / "csrc"),
                        str(source), "-o", str(exe)], check=True, timeout=120)
        layouts.append(subprocess.check_output([str(exe)], text=True))
    assert all(v == layouts[4] for v in layouts[5:])
    assert all(row.startswith("6,8,") for row in layouts[4].splitlines())
    assert all(value == layouts[0] for value in layouts[:4])  # no lost payload lanes at the target H/K


@pytest.mark.parametrize("mode", [11, 15])
def test_production_aggregate_prefetches_before_previous_submit(tmp_path, mode):
    """Execute the real Aggregate body, including its input-ring events."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG")
    arithmetic = (ROOT/"tests/ll_rdma/combine_early_reduce_test.cpp").read_text(encoding="utf-8")
    arithmetic = arithmetic[:arithmetic.index("inline uint8_t* packets=")] + "}\n"
    text = (ROOT/"csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    start = text.index("    __aicore__ inline void PreloadInputs(")
    end = text.index("    __aicore__ inline void Send(",start)
    source = arithmetic + r'''
#include "kernels/combine/low_latency/combine_profile_device.h"
using namespace AscendC;
constexpr unsigned kInputs=DEEPEP_LL_COMBINE_READY==15?6U:3U;
unsigned loaded=0, expected=0, submits=0;
struct Batch {
    bool pending;
    bool HasStaged()const{return pending;}
    void SubmitPending(){
        assert(pending && loaded==expected);
        for(unsigned lane=0;lane<expected;++lane)assert(events[int(HardEvent::MTE2_V)][lane]);
        pending=false;++submits;
    }
};
struct Partial {
    Batch* rdmaBatch_;
    CombineProfile::View profile_;
    unsigned inputLanes_=kInputs,inputSequence_=0,inputBytes_=32,h_=4;
    uint16_t input[6*16]{},low[16]{};
    float cast[16]{},product[16]{},sum[16]{};
    uint32_t table[16]{};
    LocalTensor<uint16_t> inputs_{input},low_{low};
    LocalTensor<float> cast_{cast},product_{product},sum_{sum};
    LocalTensor<uint32_t> table3Ub_{table};
    void Load(uint32_t,uint32_t,uint32_t member){
        const auto lane=(inputSequence_+member)%inputLanes_;
        WaitFlag<HardEvent::V_MTE2>(lane);
        SetFlag<HardEvent::MTE2_V>(lane);++loaded;
    }
''' + text[start:end] + r'''
};
int main(){
    for(unsigned n:{1U,2U,3U,7U})for(bool pending:{false,true}){
        loaded=submits=0;expected=std::min(n,kInputs);
        Batch batch{pending};Partial partial{&batch};
        for(unsigned lane=0;lane<kInputs;++lane)SetFlag<HardEvent::V_MTE2>(lane);
        partial.Aggregate<uint16_t>(0,0,n);
        assert(!batch.pending && submits==unsigned(pending) && loaded==n);
        for(unsigned lane=0;lane<kInputs;++lane)WaitFlag<HardEvent::V_MTE2>(lane);
        for(auto& direction:events)for(bool live:direction)assert(!live);
    }
}
'''
    cpp, exe = tmp_path/"prefetch.cpp", tmp_path/"prefetch.exe"
    cpp.write_text(source,encoding="utf-8")
    subprocess.run([zig,"c++","-std=c++17","-O1","-UNDEBUG",f"-DDEEPEP_LL_COMBINE_READY={mode}",
                    "-I"+str(ROOT/"tests/ll_rdma"),"-I"+str(ROOT/"tests/ll_rdma/stubs"),
                    "-I"+str(ROOT/"csrc"),str(cpp),"-o",str(exe)],
                   check=True,timeout=120)
    subprocess.run([str(exe)],check=True,timeout=30)


@pytest.mark.parametrize("mode", [12, 13, 14, 15, 16, 17])
def test_wide_input_aggregate_order_and_ring_reuse(tmp_path, mode):
    """Execute production preload/weighted reduction with sparse early reads."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG")
    arithmetic = (ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp").read_text(encoding="utf-8")
    arithmetic = arithmetic[:arithmetic.index("inline uint8_t* packets=")] + "}\n"
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    body = text[text.index("    __aicore__ inline void PreloadInputs("):text.index("    __aicore__ inline void Send(")]
    body = body.replace("Add(sum_", "VectorChecks::Add(sum_")
    cpp, exe = tmp_path / "inputs.cpp", tmp_path / "inputs.exe"
    cpp.write_text(arithmetic + r'''
#include "kernels/combine/low_latency/combine_profile_device.h"
#include "vector_pipeline_checks.hpp"
using namespace AscendC;
unsigned inputWaits=0;
template<HardEvent E> void CheckedWait(unsigned lane) {
    if constexpr(E==HardEvent::MTE2_V) {
        if(++inputWaits==2) {
#if DEEPEP_LL_COMBINE_READY == 16 || DEEPEP_LL_COMBINE_READY == 17
            assert(VectorChecks::multiplies==1); // first product already issued
#elif DEEPEP_LL_COMBINE_READY == 14
            assert(VectorChecks::multiplies==0); // pair waits before first product
#endif
        }
    }
    AscendC::WaitFlag<E>(lane);
}
#define WaitFlag CheckedWait
#define Cast VectorChecks::Cast
#define Muls VectorChecks::Muls
#define PipeBarrier VectorChecks::Barrier
struct Batch { bool HasStaged()const{return false;} void SubmitPending(){assert(false);} } batch;
template<class DataT> struct Partial {
    Batch* rdmaBatch_=&batch;
    CombineProfile::View profile_;
    uint32_t inputLanes_=6,inputSequence_=0,inputBytes_=32,h_=4;
    uint32_t prefetchTask_=0,prefetchMask_=0U;
    uint16_t input[6*3584]{},low[3584]{};
    float cast[3584]{},product[3584]{},sum[3584]{};
    uint32_t table[64]{},reads[32]{};
    DataT data[32][3584];
    LocalTensor<uint16_t> inputs_{input},low_{low};
    LocalTensor<float> cast_{cast},product_{product},sum_{sum};
    LocalTensor<uint32_t> table3Ub_{table};
    void Load(uint32_t,uint32_t,uint32_t member) {
        assert(++reads[member]==1);
#if DEEPEP_LL_COMBINE_READY == 17
        if(member)assert(VectorChecks::multiplies>=1); // anchor product before other loads
#endif
        const uint32_t lane=(inputSequence_+member)%inputLanes_;
        WaitFlag<HardEvent::V_MTE2>(lane);
        std::memcpy(input+lane*inputBytes_/2,data[member],h_*2);
        SetFlag<HardEvent::MTE2_V>(lane);
    }
''' + body + r'''
};
#undef WaitFlag
#undef Cast
#undef Muls
#undef PipeBarrier
template<class T> void Check(unsigned h,unsigned n,unsigned lanes,unsigned seq,unsigned mask) {
    Partial<T> p;p.inputSequence_=seq;p.inputLanes_=lanes;p.h_=h;p.inputBytes_=(h*2+31)/32*32;
    float expected[3584]{};
    for(unsigned m=0;m<n;++m) {
        float weight=(m%3U==0U?-1.f:1.f)*(0.19f+float(m)*0.013f);
        if(n>=3 && m<3)weight=1.f;
        std::memcpy(p.table+m*2U+1U,&weight,4);
        for(unsigned x=0;x<h;++x) {
            p.data[m][x]=T(float((m*7U+x*19U)%53U)-26.25f);
            // Reassociation is observable for both FP16 and BF16 inputs.
            if(n>=3 && x==0)p.data[m][x]=T(m==0?65504.f:m==1?0.001f:m==2?-65504.f:0.f);
            const float product=float(p.data[m][x])*weight;
            expected[x]=m==0 ? product : expected[x]+product;
        }
    }
    for(unsigned lane=0;lane<lanes;++lane)SetFlag<HardEvent::V_MTE2>(lane);
#if DEEPEP_LL_COMBINE_READY == 13
    p.prefetchMask_=mask&((1U<<std::min(n,lanes))-1U);
    for(unsigned m=0;m<std::min(n,lanes);++m)if(p.prefetchMask_&(1U<<m))p.Load(0,0,m);
#endif
    assert(VectorChecks::pending.empty());VectorChecks::barriers=0;VectorChecks::multiplies=0;inputWaits=0;
    p.template Aggregate<T>(0,0,n);
    assert(VectorChecks::pending.empty());
    assert(VectorChecks::barriers == (DEEPEP_LL_COMBINE_READY==16 ? 2*n+1 : 3*n-(DEEPEP_LL_COMBINE_READY==14 ? 2*(n/2):0)));
    for(unsigned x=0;x<h;++x){T value=T(expected[x]);uint16_t bits;std::memcpy(&bits,&value,2);assert(p.low[x]==bits);}
    for(unsigned m=0;m<n;++m)assert(p.reads[m]==1);
    assert(p.inputSequence_==seq+n && p.prefetchMask_==0);
    for(unsigned lane=0;lane<lanes;++lane)WaitFlag<HardEvent::V_MTE2>(lane);
    for(auto& direction:events)for(bool live:direction)assert(!live);
}
int main() {
    for(unsigned lanes=1;lanes<=6;++lanes)for(unsigned seq=0;seq<lanes;++seq)
    for(unsigned n:{1U,2U,5U,6U,7U,8U,16U,32U})for(unsigned mask:{0U,1U,0x2aU,0x3fU}) {
        Check<half>(4,n,lanes,seq,mask);Check<bfloat16_t>(4,n,lanes,seq,mask);
    }
    for(unsigned lanes:{1U,3U,6U})for(unsigned seq:{0U,1U})for(unsigned n:{1U,2U,7U,8U,32U}) {
        Check<half>(3584,n,lanes,seq,0);Check<bfloat16_t>(3584,n,lanes,seq,0);
    }
}
''', encoding="utf-8")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-ffp-contract=off", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}",
                    "-I" + str(ROOT / "tests/ll_rdma"), "-I" + str(ROOT / "tests/ll_rdma/stubs"),
                    "-I" + str(ROOT / "tests/ll_cpp_bench"),
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=60)


@pytest.mark.parametrize("mode", [12, 13, 14, 15, 16, 17])
def test_wide_input_m5_retirement(tmp_path, mode):
    """Shifted M4 output lanes must retire before M5 reuses their UB."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG")
    (tmp_path / "shmem.h").write_text("#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    exe = tmp_path / "m5.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-ffp-contract=off", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}",
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp"), "-o", str(exe)],
                   check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=120)
