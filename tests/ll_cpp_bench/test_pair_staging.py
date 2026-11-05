"""Execute the production staging batch with delayed copies and NIC reads."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("physical", [4, 16])
@pytest.mark.parametrize("profile", [False, True])
def test_pair_staging_lifetimes(tmp_path, physical, profile):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the asynchronous transport model")
    (tmp_path / "shmem.h").write_text("#pragma once\ninline void dcci_cachelines(uint8_t*, uint64_t) {}\n")
    exe = tmp_path / "pair.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    "-DDEEPEP_LL_COMBINE_READY=5", f"-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={physical}",
                    *(["-DDEEPEP_LL_COMBINE_PROFILE=1"] if profile else []),
                    "-I" + str(tmp_path), "-I" + str(ROOT / "tests/ll_rdma/stubs"),
                    "-I" + str(ROOT / "csrc"), str(Path(__file__).with_name("pair_staging_test.cpp")),
                    "-o", str(exe)], check=True, timeout=120)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=60)
    assert "PASS 288 paired staging cases" in result.stdout
