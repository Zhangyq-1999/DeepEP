"""Run production transfer adapters with delayed MTE/NIC and SDK contract stubs."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("mode", [9, 10, 11])
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
        const unsigned end=p.phaseOffset+metadata+15*480+((DEEPEP_LL_COMBINE_READY==9 || DEEPEP_LL_COMBINE_READY==11)?128:0);
        assert(end<=p.inputOffset);
        auto final=EpCombineM5Layout(p,100);
        assert(final.inputLanes);
        std::cout<<p.inputLanes<<','<<p.outputLanes<<','<<final.inputLanes<<'\n';
    }
}
''')
    layouts = []
    for mode in (4, 9, 10, 11):
        exe = tmp_path / f"layout-{mode}.exe"
        subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                        f"-DDEEPEP_LL_COMBINE_READY={mode}", "-I" + str(ROOT / "csrc"),
                        str(source), "-o", str(exe)], check=True, timeout=120)
        layouts.append(subprocess.check_output([str(exe)], text=True))
    assert all(value == layouts[0] for value in layouts)  # no lost payload lanes at the target H/K


def test_production_aggregate_prefetches_before_previous_submit(tmp_path):
    """Execute the real Aggregate body, including its input-ring events."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG")
    arithmetic = (ROOT/"tests/ll_rdma/combine_early_reduce_test.cpp").read_text(encoding="utf-8")
    arithmetic = arithmetic[:arithmetic.index("inline uint8_t* packets=")] + "}\n"
    text = (ROOT/"csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    start = text.index("    template <class T>\n    __aicore__ inline void Aggregate(")
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
