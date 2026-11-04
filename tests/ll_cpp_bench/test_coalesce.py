"""Execute production frame packing/unpacking with the Host transport model."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("rows", [0, 1])
def test_pack_transport(tmp_path, mode, rows):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    exe = tmp_path / "coalesce.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_LL_RDMA=1",
        f"-DDEEPEP_LL_COMBINE_COALESCE={mode}", f"-DDEEPEP_LL_COMBINE_COALESCE_ROWS={rows}",
        "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"), "-I" + str(ROOT / "tests/ll_rdma/stubs"),
        "-I" + str(ROOT / "tests/ll_rdma"), "-I" + str(ROOT / "csrc"),
        str(ROOT / "tests/ll_cpp_bench/coalesce_test.cpp"), "-o", str(exe)], check=True, timeout=120)
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "byte-exact unpack: PASS" in result.stdout
