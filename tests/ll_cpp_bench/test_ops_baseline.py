"""Metadata/transport adaptation checks. Device execution requires CANN/NPU."""
import os
import json
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


@pytest.mark.parametrize("profile", [False, True])
def test_ops_shmem_transport(tmp_path, profile):
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
                    *(["-DDEEPEP_LL_COMBINE_PROFILE"] if profile else []),
                    "-I", str(sdk), "-I", str(stubs),
                    str(HERE / "ops_shmem/transport_test.cpp"), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


@pytest.mark.parametrize("profile", [False, True])
def test_pinned_ops_adaptation_changes_only_transport(tmp_path, profile):
    source = os.environ.get("DEEPEP_TEST_OPS_SOURCE")
    cmake = shutil.which("cmake")
    if not source or not cmake:
        pytest.skip("Set DEEPEP_TEST_OPS_SOURCE to the pinned upstream checkout")
    source = Path(source)
    wrapper = tmp_path / "source.cmake"
    wrapper.write_text(f'set(DEEPEP_SOURCE_DIR "{HERE.parents[1].as_posix()}")\n'
                       f'set(DEEPEP_LL_COMBINE_PROFILE {"ON" if profile else "OFF"})\n'
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
    if profile:
        # Removing diagnostic-only additions must recover the clean algorithm.
        # The only functional-looking replacement caches the identical predicate
        # once to count misses; no retries, batching or ordering are changed.
        import re
        stripped = after
        stripped = stripped.replace("        const bool profileReady = IsReceiveSlotReady(source);\n", "")
        stripped = stripped.replace("if (!profileReady)", "if (!IsReceiveSlotReady(source))")
        stripped = re.sub(r"    __aicore__ inline void SetProfile\(CombineProfile::DeviceProfile\* profile\)\n"
                          r"    \{.*?    \}\nprivate:\n    CombineProfile::View profile_;\npublic:\n", "", stripped, flags=re.S)
        stripped = "\n".join(line for line in stripped.split("\n") if not any(text in line for text in
            ("profile_.", "const auto profile")))
        assert stripped == expected
        assert "CombineProfile::RdmaFirstPut" not in after  # real Put markers live in transport.hpp
        assert "CombineProfile::ReadyBegin" in after and "CombineProfile::FinalLastStore" in after
    else:
        assert after == expected
    assert (tmp_path / "ops-kernel/LICENSE").read_bytes() == (source / "LICENSE").read_bytes()


@pytest.mark.parametrize("profile", [False, True])
def test_ops_device_profile_macro_reaches_real_translation_unit(tmp_path, profile):
    zig, cmake = os.environ.get("DEEPEP_TEST_ZIG"), shutil.which("cmake")
    if not zig or not cmake or not shutil.which("ninja"):
        pytest.skip("Zig, CMake and Ninja required")
    root = HERE.parents[1]
    cann, shmem = tmp_path / "cann", tmp_path / "shmem"
    for sdk, libraries in ((cann, ("ascendcl", "runtime")), (shmem, ("shmem",))):
        for directory in ("lib", "lib64"):
            (sdk / directory).mkdir(parents=True)
            for library in libraries:
                (sdk / directory / f"lib{library}.so").touch()
    (shmem / "deepep-sdk.json").write_text(json.dumps({"build": {
        "source_patches": [], "target": "Ascend950", "cann_version": "test",
        "cann_timestamp": "test", "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": 1}}))
    build = tmp_path / "build"
    subprocess.run([cmake, "-S", str(root / "cmake/device"), "-B", str(build), "-G", "Ninja",
                    f"-DCMAKE_CXX_COMPILER={zig}", "-DCMAKE_CXX_COMPILER_ARG1=c++",
                    "-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_CXX_COMPILER_FORCED=ON",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DDEEPEP_SOURCE_DIR={root}",
                    f"-DASCEND_HOME_PATH={cann}", f"-DSHMEM_ROOT={shmem}",
                    "-DDEEPEP_LL_OPS_BASE=ON", f"-DLL_OPS_KERNEL_DIR={tmp_path}",
                    f"-DDEEPEP_LL_COMBINE_PROFILE={'ON' if profile else 'OFF'}"],
                   check=True, capture_output=True, text=True, timeout=90)
    commands = json.loads((build / "compile_commands.json").read_text())
    plain = [c for c in commands if "deepep_kernel_ll_plain.dir" in c["command"]]
    assert len(plain) == 1 and plain[0]["file"].replace("\\", "/").endswith("ops_shmem/device.cpp")
    assert ("DEEPEP_LL_COMBINE_PROFILE=1" in plain[0]["command"]) == profile
