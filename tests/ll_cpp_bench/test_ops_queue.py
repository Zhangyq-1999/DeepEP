"""Real experimental adapter + pinned SHMEM wire/DB code + asynchronous NIC model."""
import os
import json
from pathlib import Path
import subprocess
import pytest

HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE.parents[1] / "dependencies.lock.json").read_text())["shmem"]["revision"]


def function(text, name):
    start = text.index("ACLSHMEM_DEVICE", text.rfind("\nACLSHMEM_DEVICE", 0, text.index(name)) + 1)
    # Include specialization/template introducers where required.
    previous = text[:start].rstrip().splitlines()[-1]
    if previous.startswith("template <"):
        start = text.rfind("template <", 0, start)
    brace = text.index("{", text.index(name, start))
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end] + "\n"


@pytest.mark.parametrize("policy", [3, 4])
@pytest.mark.parametrize("profile", [False, True])
def test_ops_queue(tmp_path, policy, profile):
    source, zig = os.environ.get("DEEPEP_TEST_SHMEM_SOURCE"), os.environ.get("DEEPEP_TEST_ZIG")
    if not source or not zig:
        pytest.skip("Set DEEPEP_TEST_SHMEM_SOURCE (pinned SDK) and DEEPEP_TEST_ZIG")
    source = Path(source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert revision == PIN
    engine = source / "src/device/gm2gm/engine"
    abi = (engine / "shmemi_device_rdma.h").read_text(encoding="utf-8")
    wire = (engine / "rdma_backends/rdma_device_backend_hns_1825.hpp").read_text(encoding="utf-8")
    # Preserve the exact SDK declarations and encoder functions in temporary
    # build artifacts. No SDK sources are vendored or patched.
    result = abi[abi.index("enum class aclshmemi_rdma_backend_t"):abi.index("ACLSHMEM_DEVICE __gm__")]
    result += (source / "src/host_device/shmemi_rdma_cqe_layout.h").read_text(encoding="utf-8")
    result += wire[wire.index("struct aclshmemi_hns_1825_wqe_ctrl_seg_t"):wire.index("template <AscendC::HardEvent EVENT>")]
    for name in ("aclshmemi_hns_1825_sync_func", "aclshmemi_hns_1825_clear_wqe_ub",
                 "aclshmemi_hns_1825_read_u32_gm", "aclshmemi_hns_1825_write_u32_gm",
                 "aclshmemi_roce_hns_1825_get_send_wqe", "aclshmemi_roce_hns_1825_fill_wqe_ctrl_seg",
                 "aclshmemi_roce_hns_1825_fill_wqe_task_seg", "aclshmemi_roce_hns_1825_fill_wqe_data_seg",
                 "aclshmemi_roce_hns_1825_write_invalid_wqebb"):
        result += function(wire, name)
    result += "constexpr bool aclshmemi_hns_1825_use_pi_on_chip=true;\n"
    result += function(wire, "void aclshmemi_roce_hns_1825_ring_sq_doorbell_impl")
    result += "template<aclshmemi_rdma_backend_t B> void aclshmemi_roce_ring_sq_doorbell(aclshmemi_rdma_sq_ctx*&,uint32_t,AscendC::LocalTensor<uint64_t>&,AscendC::LocalTensor<uint32_t>&,uint32_t);\n"
    result += function(wire, "void aclshmemi_roce_ring_sq_doorbell")
    common = (engine / "rdma_backends/rdma_device_backend_base.hpp").read_text(encoding="utf-8")
    result += function(common, "void aclshmemi_roce_write_ub_to_gm_with_sync")
    (tmp_path / "sdk_wire.hpp").write_text(result, encoding="utf-8")
    prepared = (HERE.parents[1] / "csrc/kernels/combine/low_latency/combine_wqe_prepare.h").read_text(encoding="utf-8")
    (tmp_path / "prepared.hpp").write_text(prepared[prepared.index("namespace CombineDedup {"):], encoding="utf-8")
    (tmp_path / "shmem.h").write_text('#pragma once\n#include "queue_test_model.hpp"\n')
    backend = tmp_path / "device/gm2gm/engine/shmem_device_rdma.h"
    backend.parent.mkdir(parents=True)
    backend.write_text('#pragma once\n#include "queue_test_model.hpp"\n')
    exe = tmp_path / ("queue.exe" if os.name == "nt" else "queue")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_TEST_DEFERRED_MTE3",
                    f"-DDEEPEP_LL_OPS_POLICY={policy}",
                    *(["-DDEEPEP_LL_COMBINE_PROFILE"] if profile else []),
                    "-I", str(tmp_path), "-I", str(HERE / "ops_shmem"),
                    "-I", str(HERE.parent / "ll_rdma/stubs"),
                    str(HERE / "ops_shmem/queue_test.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)
