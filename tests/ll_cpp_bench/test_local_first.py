"""Run production M5/M6 against delayed flags and a fixed local-first golden."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile", [False, True])
def test_local_first_reduction(tmp_path, profile):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU execution")
    exe = tmp_path / "local-first.exe"
    (tmp_path / "shmem.h").write_text(
        "#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    flags = ["-DDEEPEP_LL_COMBINE_LOCAL_FIRST=1"]
    if profile:
        flags.append("-DDEEPEP_LL_COMBINE_PROFILE=1")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-ffp-contract=off", "-UNDEBUG", *flags,
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(ROOT / "tests/ll_rdma/combine_early_reduce_test.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=120)


def test_local_first_build_isolation():
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "bash"
    command = [bash, str(ROOT / "tests/ll_cpp_bench/build.sh")]
    for args, message in [
        (["--variant", "base", "--local-first", "ON"], "Use --variant opt"),
        (["--local-first", "ON", "--early-reduce", "ON"], "cannot combine"),
        (["--local-first", "ON", "--progress-probe", "ON"], "cannot combine"),
        (["--local-first", "ON", "--simt-flag", "ON"], "cannot combine"),
        (["--local-first", "bad"], "must be ON or OFF"),
    ]:
        result = subprocess.run(command + args, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and message in result.stderr
