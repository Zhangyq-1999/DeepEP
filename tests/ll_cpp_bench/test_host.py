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
    for script in ("build.sh", "run.sh", "launch.sh", "prepare_shmem.sh"):
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


def test_per_node_launch_arguments(bash):
    for ep in (2, 4, 8, 16):
        result = subprocess.run([bash, str(HERE / "launch.sh"), "--ep", str(ep), "--variants", "base,opt",
                                 "--dry-run"], capture_output=True, text=True, check=True, timeout=10)
        assert result.stdout.count(f"-ppn {ep} -n {ep}") == 2
        assert "-launcher fork" in result.stdout
        assert "0-base" in result.stdout and "1-opt" in result.stdout
    for node in (0, 1):
        result = subprocess.run([bash, str(HERE / "launch.sh"), "--ep", "32", "--hosts", "node0,node1",
                                 "--node-rank", str(node), "--master-port", "31000", "--variants", "base,opt",
                                 "--run-id", "trial", "--dry-run", "--", "--tokens", "72"],
                                capture_output=True, text=True, check=True, timeout=10)
        assert f"--proxy-id {node}" in result.stdout
        assert "--control-port node0:31000" in result.stdout
        assert "--control-port node0:31001" in result.stdout
        assert ("-n 32" in result.stdout) == (node == 0)
        if node == 0:
            assert "HYDRA_LAUNCHER_AUTOFORK=0" in result.stdout
            assert "MPIEXEC_PORT_RANGE=31000:31000" in result.stdout
    for args in (("--ep", "4", "--node-rank", "1"), ("--ep", "32"), ("--run-id", "../escape"),
                 ("--master-port", "65535", "--variants", "base,opt"), ("--variants", "unknown")):
        result = subprocess.run([bash, str(HERE / "launch.sh"), *args, "--dry-run"],
                                capture_output=True, timeout=10)
        assert result.returncode != 0


@pytest.mark.parametrize("failure", [False, True])
def test_two_node_script_sequence_with_fake_hydra(bash, tmp_path, failure):
    """Exercise two independent scripts. This fixture is not an MPI runtime test."""
    root = tmp_path / "project"
    script = root / "tests/ll_cpp_bench/launch.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "launch.sh", script)
    mpi = tmp_path / "mpi"
    (mpi / "bin").mkdir(parents=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    def write(path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, newline="\n")
        path.chmod(0o755)
    for variant in ("base", "opt"):
        write(root / f"build/ll-cpp-{variant}/ll_combine_bench", "exit 0\n")
    # Git Bash lacks setsid on some Windows installations. Process-group and
    # signal behavior belongs to Linux validation; this test checks orchestration.
    helpers = tmp_path / "helpers"
    write(helpers / "setsid", 'exec "$@"\n')
    write(helpers / "timeout", 'shift 3\nexec "$@"\n')
    write(mpi / "bin/mpiexec", r'''
if [[ ${1:-} == -version ]]; then echo '    Version: 3.2.1'; exit 0; fi
[[ $HYDRA_LAUNCHER_AUTOFORK == 0 ]]
port=${MPIEXEC_PORT_RANGE%:*}
while (($#)); do
    if [[ $1 == --output-dir ]]; then out=$2; break; fi
    shift
done
mkdir -p "$out"
touch "$FAKE_SHARED/controller-$port"
echo HYDRA_LAUNCH_END
for ((i=0;i<200;i++)); do
    if [[ -f "$FAKE_SHARED/join-$port-0" && -f "$FAKE_SHARED/join-$port-1" ]]; then break; fi
    sleep 0.02
done
[[ -f "$FAKE_SHARED/join-$port-0" && -f "$FAKE_SHARED/join-$port-1" ]]
if [[ ${FAKE_FAIL:-0} == 1 ]]; then touch "$FAKE_SHARED/done-$port"; exit 17; fi
for ((r=0;r<32;r++)); do echo passed >"$out/rank-$r.PASS"; done
echo summary >"$out/summary.csv"
touch "$FAKE_SHARED/done-$port"
''')
    write(mpi / "bin/hydra_pmi_proxy", r'''
while (($#)); do
    case $1 in --control-port) port=${2##*:};; --proxy-id) node=$2;; esac
    shift 2
done
for ((i=0;i<200;i++)); do
    [[ -f "$FAKE_SHARED/controller-$port" ]] && break
    sleep 0.02
done
[[ -f "$FAKE_SHARED/controller-$port" ]]
touch "$FAKE_SHARED/join-$port-$node"
for ((i=0;i<200;i++)); do
    [[ -f "$FAKE_SHARED/done-$port" ]] && exit 0
    sleep 0.02
done
exit 1
''')
    env = dict(os.environ, FAKE_SHARED=shared.as_posix(), FAKE_FAIL=str(int(failure)))
    # Git Bash accepts Windows drive paths as arguments, but PATH uses /c/... .
    helper_path = subprocess.check_output([bash, "-c", 'cygpath -u "$1"' if os.name == "nt" else 'printf "%s" "$1"',
                                          "_", helpers.as_posix()], text=True).strip()
    original_path = subprocess.check_output([bash, "-c", 'printf "%s" "$PATH"'], text=True)
    env["PATH"] = helper_path + ":" + original_path
    args = [bash, str(script), "--ep", "32", "--hosts", "node0,node1", "--mpi-home", mpi.as_posix(),
            "--variants", "base,opt", "--run-id", "test", "--join-timeout", "5", "--job-timeout", "10"]
    processes = [subprocess.Popen([*args, "--node-rank", str(node)], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) for node in (1, 0)]
    try:
        for proc in processes:
            output, _ = proc.communicate(timeout=25)
            assert (proc.returncode == 0) != failure, output
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
    assert (shared / "controller-29671").exists()
    assert (shared / "controller-29672").exists() != failure
