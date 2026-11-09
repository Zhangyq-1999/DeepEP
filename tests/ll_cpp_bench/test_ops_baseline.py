"""Metadata/transport adaptation checks. Device execution requires CANN/NPU."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_ops_fixture_routes_and_gather_masks(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    exe = tmp_path / ("fixture.exe" if os.name == "nt" else "fixture")
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    str(HERE / "ops_shmem/fixture_test.cpp"), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=180)


def test_ops_shmem_transport(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    # Model the SDK's header dependency as well as its transfer contract:
    # the RDMA backend consumes helpers supplied by the shmem.h include chain.
    # Compile the real adapter in isolation so a transitive include cannot hide
    # a missing prerequisite (the Device compiler reported dcci_cachelines).
    stubs = HERE.parent / "ll_rdma/stubs"
    sdk = tmp_path / "sdk"
    backend = sdk / "device/gm2gm/engine/shmem_device_rdma.h"
    backend.parent.mkdir(parents=True)
    (sdk / "shmem.h").write_text(
        "#pragma once\n#include <cstdint>\n"
        "inline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    backend.write_text(
        "#pragma once\n"
        "inline void rdma_cache_dependency(uint8_t* p) { dcci_cachelines(p, 8); }\n"
        f'#include "{(stubs / "device/gm2gm/engine/shmem_device_rdma.h").as_posix()}"\n')
    exe = tmp_path / ("transport.exe" if os.name == "nt" else "transport")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    "-I", str(sdk), "-I", str(stubs),
                    str(HERE / "ops_shmem/transport_test.cpp"), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_pinned_ops_adaptation_changes_only_transport(tmp_path):
    source = os.environ.get("DEEPEP_TEST_OPS_SOURCE")
    cmake = shutil.which("cmake")
    if not source or not cmake:
        pytest.skip("Set DEEPEP_TEST_OPS_SOURCE to the pinned upstream checkout")
    source = Path(source)
    wrapper = tmp_path / "source.cmake"
    wrapper.write_text(f'set(DEEPEP_SOURCE_DIR "{HERE.parents[1].as_posix()}")\n'
                       f'set(LL_OPS_SOURCE_DIR "{source.as_posix()}" CACHE PATH "")\n'
                       f'include("{(HERE / "ops_shmem/source.cmake").as_posix()}")\n')
    subprocess.run([cmake, "-P", str(wrapper)], cwd=tmp_path, check=True, timeout=60)
    rel = Path("mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h")
    before = (source / rel).read_text(encoding="utf-8")
    after = (tmp_path / "ops-kernel" / rel).read_text(encoding="utf-8")
    replacements = (
        ('#include "adv_api/hcomm/hcomm.h"', '#include "transport.hpp"'),
        ('Hcomm<CommProtocol::COMM_PROTOCOL_ROCE>', 'opsbench::Transport'),
        ('BatchHandle<ChannelHandle>', 'opsbench::Batch'),
        ('const ChannelHandle channel', 'const uint64_t channel'),
    )
    expected = before
    for old, new in replacements:
        assert old in expected
        expected = expected.replace(old, new)
    assert after == expected
    assert (tmp_path / "ops-kernel/LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
