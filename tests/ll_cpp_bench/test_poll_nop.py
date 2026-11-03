"""Exercise production waits with delayed arrivals; no NPU timing claims."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", ["", "PROGRESS_PROBE", "EARLY_REDUCE"])
def test_failed_probe_pause_preserves_arrival_and_reduction(tmp_path, path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU checks")
    exe = tmp_path / "poll.exe"
    (tmp_path / "shmem.h").write_text(
        "#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    flags = ["-DDEEPEP_LL_COMBINE_POLL_NOP=800", "-DDEEPEP_LL_COMBINE_PROFILE=1"]
    if path:
        flags += ["-DDEEPEP_LL_COMBINE_" + path + "=1"]
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", *flags,
                    "-I" + str(tmp_path),
                    "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=60)


def test_original_rejects_pause_and_validates_count():
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "bash"
    command = [bash, str(ROOT / "tests/ll_cpp_bench/build.sh")]
    result = subprocess.run(command + ["--variant", "base", "--poll-nop", "800"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Use --variant opt" in result.stderr
    for value in ("-1", "800.0", "ON", "0800", "65536"):
        result = subprocess.run(command + ["--poll-nop", value], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and "poll-nop must be in" in result.stderr
