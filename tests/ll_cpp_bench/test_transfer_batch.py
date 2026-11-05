"""Run production transfer adapters with delayed MTE/NIC and SDK contract stubs."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("mode", [9, 10])
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
        const unsigned end=p.phaseOffset+metadata+15*480+(DEEPEP_LL_COMBINE_READY==9?128:0);
        assert(end<=p.inputOffset);
        auto final=EpCombineM5Layout(p,100);
        assert(final.inputLanes);
        std::cout<<p.inputLanes<<','<<p.outputLanes<<','<<final.inputLanes<<'\n';
    }
}
''')
    layouts = []
    for mode in (4, 9, 10):
        exe = tmp_path / f"layout-{mode}.exe"
        subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                        f"-DDEEPEP_LL_COMBINE_READY={mode}", "-I" + str(ROOT / "csrc"),
                        str(source), "-o", str(exe)], check=True, timeout=120)
        layouts.append(subprocess.check_output([str(exe)], text=True))
    assert layouts[0] == layouts[1] == layouts[2]  # no lost payload lanes at the target H/K
