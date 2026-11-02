"""CPU arithmetic/event checks; these do not replace CANN or NPU validation."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("early", [False, True])
def test_production_early_reduction(tmp_path, early):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU C++ execution")
    exe = tmp_path / "early.exe"
    flags = ["-DDEEPEP_LL_COMBINE_EARLY_REDUCE=1"] if early else []
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-ffp-contract=off", "-UNDEBUG", *flags,
                    "-I" + str(ROOT / "csrc"), "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    str(ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=60)


@pytest.mark.parametrize("early", [False, True])
def test_early_reduce_host_query_without_device_helpers(tmp_path, early):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU C++ execution")
    text = (ROOT / "csrc/kernels/combine/low_latency.cpp").read_text(encoding="utf-8")
    start = text.index('extern "C" DEEPEP_EXPORT uint32_t deepep_ll_combine_early_reduce_enabled()')
    query = text[start:text.index("\n}", start) + 2]
    # Compile the actual exported Host query without any AICore helpers.
    source = tmp_path / "query.cpp"
    source.write_text("#include <cstdint>\n#define DEEPEP_EXPORT\n" + query +
                      f"\nint main() {{ return deepep_ll_combine_early_reduce_enabled() != {int(early)}U; }}\n",
                      encoding="utf-8")
    exe = tmp_path / "query.exe"
    flags = ["-DDEEPEP_LL_COMBINE_EARLY_REDUCE=1"] if early else []
    subprocess.run([zig, "c++", "-std=c++17", *flags, str(source), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_tuning_build_identity():
    spec = importlib.util.spec_from_file_location("tuning_build", ROOT / "build_support.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    assert build.combine_recv_cores({}) == 0
    for value in (0, 8, 16, 24, 32, 36):
        assert build.combine_recv_cores({"DEEPEP_LL_COMBINE_RECV_CORES": str(value)}) == value
    for value in ("-1", "7", "37", "8.0", "auto", ""):
        with pytest.raises(ValueError):
            build.combine_recv_cores({"DEEPEP_LL_COMBINE_RECV_CORES": value})
    keys = {build.build_cache_key({"ll_combine_recv_cores": count, "ll_combine_early_reduce": early})
            for count in (0, 8, 16, 24, 32) for early in (False, True)}
    assert len(keys) == 10
