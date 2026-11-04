"""Exercise the shared build graph and real source selection without CANN."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(*args):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def cmake():
    tool = shutil.which("cmake")
    if not tool or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja required")
    return tool


def test_shared_objects_incremental_and_configuration_isolation(tmp_path, cmake):
    root = tmp_path / "source with spaces"
    child = root / "cmake/device"
    child.mkdir(parents=True)
    cann, shmem = tmp_path / "cann", tmp_path / "shmem"
    compiler = cann / "tools/bisheng_compiler/bin/bisheng"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("compiler identity")
    shmem.mkdir()
    (shmem / "deepep-sdk.json").write_text("SDK identity")
    source, header, combine = root / "dispatch.cpp", root / "dispatch.h", root / "combine.cpp"
    for f in (source, header, combine):
        f.write_text("original")
    (child / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(dispatch_fixture NONE)
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/dispatch.o"
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/dispatch.o"
    DEPENDS "${DEEPEP_SOURCE_DIR}/dispatch.cpp" "${DEEPEP_SOURCE_DIR}/dispatch.h")
add_custom_target(objects ALL DEPENDS "${CMAKE_BINARY_DIR}/dispatch.o")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/dispatch-objects.txt" CONTENT "${CMAKE_BINARY_DIR}/dispatch.o\\n")
''')
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.20)
project(variant_fixture NONE)
set(DEEPEP_SOURCE_DIR "{root.as_posix()}")
set(ASCEND_HOME_PATH "{cann.as_posix()}")
set(SHMEM_ROOT "{shmem.as_posix()}")
set(DEEPEP_LL_METADATA_CORES 4)
set(DEEPEP_DEVICE_JOBS 2)
include("{(HERE / 'shared_dispatch.cmake').as_posix()}")
ll_bench_shared_dispatch()
file(WRITE "${{CMAKE_BINARY_DIR}}/object-list.txt" "${{DEEPEP_DISPATCH_OBJECTS_FILE}}")
''')

    def build(name, ranks=4):
        dest = tmp_path / name
        run(cmake, "-S", parent, "-B", dest, "-G", "Ninja",
            f"-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={ranks}")
        run(cmake, "--build", dest, "--target", "ll_bench_dispatch_build")
        objects = Path((dest / "object-list.txt").read_text())
        return Path(objects.read_text().strip())

    obj = build("base")
    stamp = obj.stat().st_mtime_ns
    assert build("issue") == obj and obj.stat().st_mtime_ns == stamp
    combine.write_text("changed Combine only")
    assert build("issue") == obj and obj.stat().st_mtime_ns == stamp
    # A real header dependency must rebuild even though the configuration key
    # remains unchanged. No source hash or manual cache clearing is required.
    header.write_text("changed Dispatch header")
    os.utime(header, ns=(stamp + 2_000_000_000, stamp + 2_000_000_000))
    assert build("issue") == obj and obj.stat().st_mtime_ns != stamp
    assert build("sixteen", 16) != obj
    (shmem / "deepep-sdk.json").write_text("new SDK identity")
    assert build("new-sdk") != obj


@pytest.mark.parametrize("entry_sync", [False, True])
def test_device_components_select_real_sources(tmp_path, cmake, entry_sync):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    cann, shmem = tmp_path / "cann", tmp_path / "shmem"
    for sdk, libraries in ((cann, ("ascendcl", "runtime")), (shmem, ("shmem",))):
        (sdk / "lib64").mkdir(parents=True)
        (sdk / "lib").mkdir()
        for lib in libraries:
            for directory in ("lib", "lib64"):
                (sdk / directory / f"lib{lib}.so").touch()
    (shmem / "deepep-sdk.json").write_text(json.dumps({"build": {
        "source_patches": [], "target": "Ascend950", "cann_version": "test",
        "cann_timestamp": "test", "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": 1}}))
    objlist = tmp_path / "dispatch-objects.txt"
    objlist.write_text((tmp_path / "shared-dispatch.o").as_posix() + "\n")
    for component, expected in (("all", 18), ("dispatch", 17), ("combine", 1)):
        build = tmp_path / component
        run(cmake, "-S", ROOT / "cmake/device", "-B", build, "-G", "Ninja",
            f"-DCMAKE_CXX_COMPILER={zig}", "-DCMAKE_CXX_COMPILER_ARG1=c++",
            "-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_CXX_COMPILER_FORCED=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DDEEPEP_SOURCE_DIR={ROOT}",
            f"-DASCEND_HOME_PATH={cann}", f"-DSHMEM_ROOT={shmem}",
            f"-DDEEPEP_DEVICE_COMPONENT={component}", f"-DDEEPEP_DISPATCH_OBJECTS_FILE={objlist}",
            f"-DDEEPEP_LL_BENCH_ENTRY_SYNC={'ON' if entry_sync else 'OFF'}")
        commands = json.loads((build / "compile_commands.json").read_text())
        plain = [c for c in commands if "deepep_kernel_ll_plain.dir" in c["command"]]
        added = int(entry_sync and component != "combine")
        assert len(plain) == expected + added
        assert sum(Path(c["file"]).name == "entry_sync.cpp" for c in plain) == added
        if component == "combine":
            assert all("/combine/" in c["file"].replace("\\", "/") for c in plain)
            assert "shared-dispatch.o" in (build / "build.ninja").read_text()
        elif component == "dispatch":
            assert all("/dispatch/" in c["file"].replace("\\", "/") or Path(c["file"]).name == "entry_sync.cpp" for c in plain)
            assert len((build / "dispatch-objects.txt").read_text().splitlines()) == 17 + added
