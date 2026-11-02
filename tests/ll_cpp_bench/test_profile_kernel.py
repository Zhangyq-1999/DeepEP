"""Compile all Combine template bodies with CPU API stubs, including diagnostics.

This catches missing bindings and locals; device compilation still requires CANN.
"""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("returns", [False, True])
def test_all_combine_profile_paths_compile(tmp_path, profile, small, returns):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU compile checks")
    # Reuse arithmetic/API declarations; no simulated synchronization changes.
    arithmetic = (ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp").read_text(encoding="utf-8")
    arithmetic = arithmetic[:arithmetic.index("inline uint8_t* packets=")] + "}\n"
    source = tmp_path / "all.cpp"
    source.write_text(arithmetic + r'''
namespace AscendC {
template<class T> void ReduceSum(LocalTensor<T>,LocalTensor<T>,LocalTensor<T>,uint32_t,uint32_t,uint32_t) {}
template<class T> void Gather(LocalTensor<T>,LocalTensor<T>,LocalTensor<uint32_t>,uint32_t,uint32_t) {}
struct ProfileCopyParams {uint32_t dstStride,srcStride,dstRepeat,srcRepeat;};
template<class T> void Copy(LocalTensor<T>,LocalTensor<T>,uint64_t,uint8_t,ProfileCopyParams) {}
}
#include "kernels/combine/low_latency/combine_server_dedup_kernel.h"
template class CombineDedup::CombineMemoryServerDedupKernel<2>;
template class CombineDedup::CombineMemoryServerDedupKernel<4>;
template class CombineDedup::CombineMemoryServerDedupKernel<16>;
int main() {}
''', encoding="utf-8")
    (tmp_path / "shmem.h").write_text('#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n')
    flags = ["-DDEEPEP_LL_RDMA=1", "-DDEEPEP_LL_COMBINE_EARLY_REDUCE=1",
             "-DDEEPEP_LL_COMBINE_RECV_CORES=36"]
    if returns:
        flags += ["-DDEEPEP_LL_COMBINE_REMOTE_FIRST=1", "-DDEEPEP_LL_COMBINE_DEFER_RETURN=1"]
    if profile:
        flags += ["-DDEEPEP_LL_COMBINE_PROFILE=1"]
    if small:
        flags += ["-DDEEPEP_LL_COMBINE_SMALL_BATCH=1"]
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-c", *flags,
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tests/ll_rdma"),
                    "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"), str(source),
                    "-o", str(tmp_path / "all.o")],
                   check=True, timeout=120)
