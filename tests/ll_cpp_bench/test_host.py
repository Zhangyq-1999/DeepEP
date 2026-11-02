"""Host-only checks; actual MPI/SHMEM/NPU execution is a separate validation."""
import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_options_routes_and_statistics(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for host C++ checks")
    executable = tmp_path / "options.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", str(HERE / "options_test.cpp"),
                    "-o", str(executable)], check=True, timeout=180)
    subprocess.run([str(executable)], check=True, timeout=30)


@pytest.fixture
def bash():
    candidate = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not candidate or not Path(candidate).exists():
        pytest.skip("Bash unavailable")
    return candidate


def test_shell_syntax(bash):
    for script in ("build.sh", "run.sh", "prepare_shmem.sh"):
        subprocess.run([bash, "-n", str(HERE / script)], check=True, timeout=10)


@pytest.mark.parametrize("nested", [False, True])
def test_sdk_manifest_record_and_check(tmp_path, nested):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake unavailable")
    sdk = tmp_path / "sdk"
    (sdk / "include").mkdir(parents=True)
    (sdk / "lib").mkdir()
    header = b"// fixture\n"
    (sdk / "include/test.h").write_bytes(header)
    (sdk / "lib/libshmem.so").write_bytes(b"sdk fixture")
    (sdk / "lib/aclshmem_bootstrap_mpi.so").write_bytes(b"mpi fixture")
    (tmp_path / "dependencies.lock.json").write_text(json.dumps({"shmem": {
        "revision": "fixture", "header_sha256_lf": {"test.h": hashlib.sha256(header).hexdigest()}}}))
    build = tmp_path / "build"
    flags = build / ("src/device/" if nested else "") / "CMakeFiles/aclshmem_device.dir/flags.make"
    flags.parent.mkdir(parents=True)
    flags.write_text("CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=1\n")
    cann = tmp_path / "cann"
    (cann / "compiler").mkdir(parents=True)
    (cann / "compiler/version.info").write_text("Version=9.3.0\ntimestamp=fixture\n")
    common = [cmake, f"-DPROJECT_ROOT={tmp_path}", f"-DSDK={sdk}", f"-DSDK_BUILD={build}",
              f"-DASCEND_HOME_PATH={cann}"]
    def run(mode, success=True):
        result = subprocess.run([*common, f"-DMODE={mode}", "-P", str(HERE / "sdk_identity.cmake")],
                                capture_output=True, text=True, timeout=10)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
    run("record")
    identity = json.loads((sdk / "deepep-sdk.json").read_text())
    assert identity["build"]["hns_1825_pi_on_chip"] == 1
    assert identity["build"]["rdma_backend"] == "HNS_1825"
    assert len(identity["sha256"]) == 3
    run("check")
    run("record", success=False)
    (sdk / "lib/aclshmem_bootstrap_mpi.so").write_bytes(b"changed")
    run("check", success=False)


def test_mpi_launch_commands(bash):
    for hosts, launcher, ranks in (("localhost", "fork", 16), ("node0,node1", "manual", 32)):
        result = subprocess.run([bash, str(HERE / "run.sh"), "--hosts", hosts,
                                 "--launcher", launcher, "--master-addr", "node0", "--dry-run", "--",
                                 "--tokens", "72", "--output-dir", "/tmp/test data"],
                                capture_output=True, text=True, timeout=10, check=True)
        assert f"-ppn 16 -n {ranks}" in result.stdout
        assert f"-launcher {launcher}" in result.stdout
        assert "--tokens 72" in result.stdout
        assert "MPIEXEC_PORT_RANGE=29671:29771" in result.stdout
    for ep in (2,4,8,16):
        result = subprocess.run([bash, str(HERE / "run.sh"), "--hosts", "localhost", "--ep", str(ep),
                                 "--launcher", "fork", "--dry-run"], capture_output=True, text=True, timeout=10, check=True)
        assert f"-ppn {ep} -n {ep}" in result.stdout
    for args in (("--hosts", "a,a"), ("--hosts", "a,b", "--launcher", "fork"),
                 ("--port-range", "3000:2000"), ("--hosts", "a,b,c"), ("--hosts", "a,b", "--ep", "8"),
                 ("--hosts", "a", "--ep", "32")):
        result = subprocess.run([bash, str(HERE / "run.sh"), *args, "--dry-run"],
                                capture_output=True, timeout=10)
        assert result.returncode != 0
