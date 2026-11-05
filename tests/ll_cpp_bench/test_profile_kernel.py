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
@pytest.mark.parametrize("returns", [False, True, "new", "simt", "poll", "local", "batch", "batch-pipeline",
                                     "card-all", "card-row", "server-all", "server-row", "notify-simt", "notify-core", "ready-task", "ready-shared", "ready-fast", "ready-pipeline", "ready-snapshot", "stage-pair", "stage-prefetch", "stage-stream", "plan-compact", "ready-r32", "ready-r36", "wqe-prep", "mte-pack", "wqe-prefetch", "input6", "input-ready", "input-pair", "input-pipe"])
def test_all_combine_profile_paths_compile(tmp_path, profile, small, returns):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU compile checks")
    # Reuse arithmetic/API declarations; no simulated synchronization changes.
    arithmetic = (ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp").read_text(encoding="utf-8")
    arithmetic = arithmetic[:arithmetic.index("inline uint8_t* packets=")] + "}\n"
    source = tmp_path / "all.cpp"
    source.write_text('#include "simt_test_stub.h"\n' + arithmetic + r'''
namespace AscendC {
template<class T> void Adds(LocalTensor<T> d,LocalTensor<T> s,T x,uint32_t n) { for(uint32_t i=0;i<n;++i)d.p[i]=s.p[i]+x; }
template<class T> void ReduceSum(LocalTensor<T>,LocalTensor<T>,LocalTensor<T>,uint32_t,uint32_t,uint32_t) {}
template<class T> void ReduceSum(LocalTensor<T>,LocalTensor<T>,LocalTensor<T>,int32_t) {}
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
    if isinstance(returns,str) and returns.startswith(("card-","server-")):
        flags = ["-DDEEPEP_LL_RDMA=1", "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4",
                 f'-DDEEPEP_LL_COMBINE_COALESCE={1 if returns.startswith("card-") else 2}',
                 f'-DDEEPEP_LL_COMBINE_COALESCE_ROWS={int(returns.endswith("-row"))}']
    elif returns in ("ready-task", "ready-shared", "ready-fast", "ready-pipeline", "ready-snapshot", "stage-pair", "stage-prefetch", "stage-stream", "plan-compact", "ready-r32", "ready-r36", "wqe-prep", "mte-pack", "wqe-prefetch", "input6", "input-ready", "input-pair", "input-pipe"):
        if small: pytest.skip("Ready modes are independent of small-batch")
        flags = ["-DDEEPEP_LL_RDMA=1", f'-DDEEPEP_LL_COMBINE_READY={1 if returns == "ready-task" else 2 if returns == "ready-shared" else 4 if returns in ("ready-snapshot", "ready-r32", "ready-r36") else 5 if returns == "stage-pair" else 6 if returns == "stage-prefetch" else 7 if returns == "stage-stream" else 8 if returns == "plan-compact" else 9 if returns == "wqe-prep" else 10 if returns == "mte-pack" else 12 if returns == "input6" else 14 if returns == "input-pair" else 15 if returns == "input-pipe" else 13 if returns == "input-ready" else 11 if returns == "wqe-prefetch" else 3}']
        if returns.startswith("ready-r"): flags.append(f"-DDEEPEP_LL_COMBINE_RECV_CORES={returns[-2:]}")
        if returns == "ready-pipeline": flags.append("-DDEEPEP_LL_COMBINE_DEFER_RETURN=1")
    elif returns in ("notify-simt", "notify-core"):
        flags += [f'-DDEEPEP_LL_COMBINE_NOTIFY={1 if returns == "notify-simt" else 2}',
                  "-DDEEPEP_LL_COMBINE_REMOTE_FIRST=1"]
    elif returns == "poll":
        flags += ["-DDEEPEP_LL_COMBINE_POLL_NOP=800", "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4"]
    elif returns in ("local", "batch", "batch-pipeline"):
        flags = ["-DDEEPEP_LL_RDMA=1", "-DDEEPEP_LL_COMBINE_LOCAL_FIRST=1",
                 "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4"]
        if returns.startswith("batch"):
            flags.append("-DDEEPEP_LL_COMBINE_BATCH_FINAL=1")
        if returns == "batch-pipeline":
            flags.append("-DDEEPEP_LL_COMBINE_PREFETCH_RETURN=1")
    elif returns == "simt":
        flags = ["-DDEEPEP_LL_RDMA=1", "-DDEEPEP_LL_COMBINE_SIMT_META=1", "-DDEEPEP_LL_COMBINE_SIMT_FLAG=1",
                 "-DDEEPEP_LL_COMBINE_PREFETCH_RETURN=1", "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4"]
    elif returns == "new":
        flags += ["-DDEEPEP_LL_COMBINE_EARLY_ISSUE=1", "-DDEEPEP_LL_COMBINE_PROGRESS_PROBE=1",
                  "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4"]
    elif returns:
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
