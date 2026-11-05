"""Run production transfer adapters with delayed MTE/NIC and SDK contract stubs."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("mode", [9, 10, 11, 12, 13])
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
    for mode in (4, 9, 10, 11, 12, 13):
        exe = tmp_path / f"layout-{mode}.exe"
        subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                        f"-DDEEPEP_LL_COMBINE_READY={mode}", "-I" + str(ROOT / "csrc"),
                        str(source), "-o", str(exe)], check=True, timeout=120)
        layouts.append(subprocess.check_output([str(exe)], text=True))
    assert layouts[4] == layouts[5]
    assert all(row.startswith("6,8,") for row in layouts[4].splitlines())
    assert all(value == layouts[0] for value in layouts[:4])  # no lost payload lanes at the target H/K


def test_production_aggregate_prefetches_before_previous_submit(tmp_path):
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
    unsigned inputLanes_=3,inputSequence_=0,inputBytes_=32,h_=4;
    uint16_t input[48]{},low[16]{};
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
        loaded=submits=0;expected=std::min(n,3U);
        Batch batch{pending};Partial partial{&batch};
        for(unsigned lane=0;lane<3;++lane)SetFlag<HardEvent::V_MTE2>(lane);
        partial.Aggregate<uint16_t>(0,0,n);
        assert(!batch.pending && submits==unsigned(pending) && loaded==n);
        for(unsigned lane=0;lane<3;++lane)WaitFlag<HardEvent::V_MTE2>(lane);
        for(auto& direction:events)for(bool live:direction)assert(!live);
    }
}
'''
    cpp, exe = tmp_path/"prefetch.cpp", tmp_path/"prefetch.exe"
    cpp.write_text(source,encoding="utf-8")
    subprocess.run([zig,"c++","-std=c++17","-O1","-UNDEBUG","-DDEEPEP_LL_COMBINE_READY=11",
                    "-I"+str(ROOT/"tests/ll_rdma"),"-I"+str(ROOT/"tests/ll_rdma/stubs"),
                    "-I"+str(ROOT/"csrc"),str(cpp),"-o",str(exe)],
                   check=True,timeout=120)
    subprocess.run([str(exe)],check=True,timeout=30)


@pytest.mark.parametrize("mode", [12, 13])
def test_wide_input_aggregate_order_and_ring_reuse(tmp_path, mode):
    """Execute production preload/weighted reduction with sparse early reads."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG")
    arithmetic = (ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp").read_text(encoding="utf-8")
    arithmetic = arithmetic[:arithmetic.index("inline uint8_t* packets=")] + "}\n"
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    body = text[text.index("    __aicore__ inline void PreloadInputs("):text.index("    __aicore__ inline void Send(")]
    cpp, exe = tmp_path / "inputs.cpp", tmp_path / "inputs.exe"
    cpp.write_text(arithmetic + r'''
#include "kernels/combine/low_latency/combine_profile_device.h"
using namespace AscendC;
struct Partial {
    CombineProfile::View profile_;
    uint32_t inputLanes_=6,inputSequence_=0,inputBytes_=32,h_=4;
    uint32_t prefetchTask_=0,prefetchMask_=0;
    uint16_t input[6*16]{},low[16]{};
    float cast[16]{},product[16]{},sum[16]{};
    uint32_t table[64]{},reads[32]{};
    bfloat16_t data[32][4];
    LocalTensor<uint16_t> inputs_{input},low_{low};
    LocalTensor<float> cast_{cast},product_{product},sum_{sum};
    LocalTensor<uint32_t> table3Ub_{table};
    void Load(uint32_t,uint32_t,uint32_t member) {
        assert(++reads[member]==1); // speculative reads must not be repeated
        const uint32_t lane=(inputSequence_+member)%inputLanes_;
        WaitFlag<HardEvent::V_MTE2>(lane);
        std::memcpy(input+lane*16,data[member],8);
        SetFlag<HardEvent::MTE2_V>(lane);
    }
''' + body + r'''
};
int main() {
    for(unsigned n:{1U,2U,5U,6U,7U,8U,16U,32U})for(unsigned seq=0;seq<6;++seq)
    for(unsigned mask:{0U,1U,0x2aU,0x3fU}) {
        Partial p;p.inputSequence_=seq;
        float expected[4]{};
        for(unsigned m=0;m<n;++m) {
            float weight=(m%3U==0U?-1.f:1.f)*(0.19f+float(m)*0.013f);
            std::memcpy(p.table+m*2U+1U,&weight,4);
            for(unsigned h=0;h<4;++h) {
                p.data[m][h]=bfloat16_t(float((m*7U+h*19U)%53U)-26.25f);
                const float product=float(p.data[m][h])*weight;
                expected[h]=m==0 ? product : expected[h]+product;
            }
        }
        for(unsigned lane=0;lane<6;++lane)SetFlag<HardEvent::V_MTE2>(lane);
#if DEEPEP_LL_COMBINE_READY == 13
        p.prefetchMask_=mask&((1U<<std::min(n,6U))-1U);
        for(unsigned m=0;m<std::min(n,6U);++m)if(p.prefetchMask_&(1U<<m))p.Load(0,0,m);
#endif
        p.Aggregate<bfloat16_t>(0,0,n);
        for(unsigned h=0;h<4;++h)assert(p.low[h]==bfloat16_t(expected[h]).bits);
        for(unsigned m=0;m<n;++m)assert(p.reads[m]==1);
        assert(p.inputSequence_==seq+n && p.prefetchMask_==0);
        for(unsigned lane=0;lane<6;++lane)WaitFlag<HardEvent::V_MTE2>(lane);
        for(auto& direction:events)for(bool live:direction)assert(!live);
    }
}
''', encoding="utf-8")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-ffp-contract=off", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}",
                    "-I" + str(ROOT / "tests/ll_rdma"), "-I" + str(ROOT / "tests/ll_rdma/stubs"),
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=30)


@pytest.mark.parametrize("mode", [12, 13])
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
