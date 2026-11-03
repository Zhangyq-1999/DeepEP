"""Exercise convenience scripts with fake builds/launches; no NPU required."""
import os
from pathlib import Path
import shutil
import shlex
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
PROFILES = ("ref-profile", "batch-profile", "batch-pipeline-profile")


@pytest.fixture
def harness(tmp_path):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash unavailable")
    root = tmp_path / "project with spaces"
    scripts = root / "tests/ll_cpp_bench"
    scripts.mkdir(parents=True)
    for name in ("build_all.sh", "run_all.sh", "experiment_defaults.sh"):
        shutil.copyfile(HERE / name, scripts / name)
    capture = r'''#!/usr/bin/env bash
set -eu
printf 'CALL\n'
printf '%s\n' "$@"
if [[ ${FAIL_BUILD:-} != '' && " $* " == *"ll-cpp-$FAIL_BUILD "* ]]; then exit 17; fi
'''
    for name in ("build.sh", "launch.sh"):
        (scripts / name).write_text(capture, newline="\n")
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    hostname = helpers / "hostname"
    hostname.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$TEST_LOCAL_IP"\n', newline="\n")
    hostname.chmod(0o755)
    env = os.environ.copy()
    for name in ("NODE_RANK", "HOST0", "HOST1", "ASCEND_HOME_PATH", "MPI_HOME", "SHMEM_ROOT"):
        env.pop(name, None)
    helper_path = subprocess.check_output([bash, "-c", 'cygpath -u "$1"' if os.name == "nt" else 'printf "%s" "$1"',
                                          "_", helpers.as_posix()], text=True).strip()
    original_path = subprocess.check_output([bash, "-c", 'printf "%s" "$PATH"'], text=True)
    env["PATH"] = helper_path + ":" + original_path
    env["TEST_LOCAL_IP"] = "90.90.93.84"

    def run(script, *args, ok=True):
        # Git Bash prepends its system directories during startup. Put the
        # hostname fixture first after startup, as a Linux PATH lookup would.
        result = subprocess.run([bash, "-c", 'export PATH="$1:$PATH"; shift; script=$1; shift; source "$script" "$@"',
                                 "_", helper_path, str(scripts / script), *args], env=env,
                                capture_output=True, text=True, timeout=15)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        return result
    return run, env


def call_options(output):
    calls = []
    for call in output.split("CALL\n")[1:]:
        lines = call.splitlines()
        values = {}
        for i, line in enumerate(lines):
            if line.startswith("--") and line != "--":
                assert line not in values, f"duplicate {line}"
                values[line] = lines[i + 1]
        calls.append(values)
    return calls


def test_build_all_isolated_matrix_and_failure_stop(harness):
    run, env = harness
    calls = call_options(run("build_all.sh").stdout)
    assert len(calls) == 6
    for name, call in zip(("base", "batch", "batch-pipeline", *PROFILES), calls):
        assert call["--build-dir"].endswith("/build/ll-cpp-" + name)
        assert call["--variant"] == ("base" if name == "base" else "opt")
        assert call["--profile"] == ("ON" if name.endswith("-profile") else "OFF")
        assert call["--small-batch"] == ("ON" if name == "opt-profile" else "OFF")
        assert call["--recv-cores"] == ("36" if name == "r36-profile" else "0")
        assert call["--early-reduce"] == ("ON" if name == "early-profile" else "OFF")
        assert call["--remote-first"] == ("ON" if name in ("remote", "return", "remote-profile", "return-profile") else "OFF")
        assert call["--defer-return"] == ("ON" if name in ("pipeline", "return", "pipeline-profile", "return-profile") else "OFF")
        assert call["--ranks-per-host"] == "4"
        assert call["--early-issue"] == ("ON" if name in ("issue", "issue-probe", "issue-profile", "issue-probe-profile") else "OFF")
        assert call["--progress-probe"] == ("ON" if name in ("probe", "issue-probe", "probe-profile", "issue-probe-profile") else "OFF")
        assert call["--simt-meta"] == ("ON" if name.startswith("prepare") else "OFF")
        assert call["--prefetch-return"] == ("ON" if name.startswith("batch-pipeline") else "OFF")
        assert call["--simt-flag"] == "OFF"
        assert call["--local-first"] == ("ON" if name.startswith("batch") else "OFF")
        assert call["--batch-final"] == ("ON" if name.startswith("batch") else "OFF")
        assert call["--poll-nop"] == ("800" if name.startswith("poll800") else "0")
        assert call["--shmem-root"].endswith("/third-party/shmem/install/shmem")
    env["FAIL_BUILD"] = "ref-profile"
    result = run("build_all.sh", ok=False)
    assert result.returncode == 17 and len(call_options(result.stdout)) == 4
    assert "CALL" not in run("build_all.sh", "--variants", "base,typo", ok=False).stdout


def test_build_sdk_and_selection_overrides(harness):
    run, env = harness
    env["SHMEM_ROOT"] = "/env/sdk"
    calls = call_options(run("build_all.sh", "--variants", "return-profile,ref-profile,opt-profile,r36-profile,early-profile,poll800,poll800-profile",
                             "--jobs", "8", "--shmem-root", "/selected/sdk with spaces",
                             "--mpi-home", "/selected/mpi", "--cann-home", "/selected/cann").stdout)
    assert len(calls) == 7
    assert calls[2]["--small-batch"] == "ON"
    assert calls[3]["--recv-cores"] == "36"
    assert calls[4]["--early-reduce"] == "ON"
    assert all(call["--poll-nop"] == "800" for call in calls[5:])
    for call in calls:
        assert call["--jobs"] == "8"
        assert call["--shmem-root"] == "/selected/sdk with spaces"
        assert call["--mpi-home"] == "/selected/mpi"
        assert call["--cann-home"] == "/selected/cann"


def test_repeat_default_run_and_second_host_join(harness):
    run, env = harness
    ids = []
    for _ in range(2):
        result = run("run_all.sh")
        call, = call_options(result.stdout)
        ids.append(call["--run-id"])
        assert call["--node-rank"] == "0" and call["--ep"] == "8"
        assert call["--devices"] == "12,13,14,15" and call["--ranks-per-host"] == "4"
        assert call["--hosts"] == "90.90.93.84,90.90.93.184"
        assert call["--variants"] == ",".join(("base", "batch", "batch-pipeline", *PROFILES))
        assert call["--tokens"] == "32,48,64,72,256"
        assert call["--iters"] == "100" and call["--profile"] == "2"
        assert call["--profile-iters"] == "10"
        assert f"run_all.sh {ids[-1]} \n" in result.stdout
    assert len(set(ids)) == 2
    env["TEST_LOCAL_IP"] = "90.90.93.184"
    joined, = call_options(run("run_all.sh", ids[0]).stdout)
    assert joined["--node-rank"] == "1" and joined["--run-id"] == ids[0]
    assert "Copy RUN_ID" in run("run_all.sh", ok=False).stderr


def test_host_detection_and_benchmark_overrides(harness):
    run, env = harness
    env["TEST_LOCAL_IP"] = "unlisted-host"
    assert "--node-rank" in run("run_all.sh", ok=False).stderr
    result = run("run_all.sh", "--node-rank", "0", "--hosts", "a,b", "--master-port", "30071",
                 "--variants", "return-profile,ref-profile", "--tokens", "72,256", "--iters", "20",
                 "--", "--warmup", "5", "--seed", "4")
    call, = call_options(result.stdout)
    assert call["--tokens"] == "72,256" and call["--iters"] == "20" and call["--warmup"] == "5"
    assert call["--master-addr"] == "a" and call["--master-port"] == "30071"
    peer_line = next(line for line in result.stdout.splitlines() if line.startswith("Run on node 1:"))
    peer_args = shlex.split(peer_line.removeprefix("Run on node 1: "))
    assert peer_args[peer_args.index("--hosts") + 1] == "a,b"
    assert peer_args[peer_args.index("--variants") + 1] == "return-profile,ref-profile"
    assert peer_args[peer_args.index("--node-rank") + 1] == "1"
    assert peer_args[peer_args.index("--master-port") + 1] == "30071"
    local, = call_options(run("run_all.sh", "--ep", "4", "--devices", "6,7,14,15").stdout)
    assert local["--hosts"] == "localhost" and local["--node-rank"] == "0"
    assert local["--devices"] == "6,7,14,15" and local["--entry-sync"] == "host"
    assert call_options(run("run_all.sh", "--ep", "16", "--ranks-per-host", "16", "--variants", "base").stdout)[0]["--profile"] == "2"
    clean, = call_options(run("run_all.sh", "--ep", "16", "--ranks-per-host", "16", "--variants", "base", "--profile", "0").stdout)
    assert clean["--profile"] == "0"
    env["TEST_LOCAL_IP"] = "90.90.93.84 90.90.93.184"
    assert "Multiple hosts match" in run("run_all.sh", ok=False).stderr


def test_four_host_commands_and_detection(harness):
    run, env = harness
    hosts = "90.90.93.84,90.90.93.184,90.90.93.85,90.90.93.185"
    result = run("run_all.sh", "--ep", "64", "--ranks-per-host", "16", "--hosts", hosts,
                 "--tokens", "32,72,128,256", "--master-port", "30071")
    first, = call_options(result.stdout)
    assert first["--devices"] == ",".join(map(str, range(16)))
    assert first["--tokens"] == "32,72,128,256"
    lines = [line for line in result.stdout.splitlines() if line.startswith("Run on node ")]
    assert len(lines) == 3
    for node, line in enumerate(lines, 1):
        command = shlex.split(line.split(": ", 1)[1])
        peer, = call_options(run("run_all.sh", *command[2:]).stdout)
        for key in ("--ep", "--hosts", "--master-addr", "--master-port", "--run-id", "--ranks-per-host", "--variants"):
            assert peer[key] == first[key]
        assert peer["--node-rank"] == str(node)
        env["TEST_LOCAL_IP"] = hosts.split(",")[node]
        detected, = call_options(run("run_all.sh", first["--run-id"], "--ep", "64",
                                    "--ranks-per-host", "16", "--hosts", hosts).stdout)
        assert detected["--node-rank"] == str(node)


@pytest.mark.parametrize("args", [
    ("--ep", "64", "--ranks-per-host", "16"),
    ("--ep", "64", "--hosts", "a,b,c,d"),
    ("--ep", "64", "--ranks-per-host", "16", "--hosts", "a,b,c,a"),
    ("--ep", "64", "--ranks-per-host", "16", "--hosts", "a,b,c,d,other"),
    ("--node-rank", "2"),
])
def test_invalid_multi_host_layout(harness, args):
    run, _ = harness
    run("run_all.sh", "--dry-run", *args, ok=False)


def test_optional_ops_baseline(harness):
    run, env = harness
    commands = call_options(run("build_all.sh", "--variants", "base,ops-shmem-base").stdout)
    assert len(commands) == 2
    assert commands[1]["--variant"] == "ops-shmem-base"
    assert commands[1]["--profile"] == "OFF"
    assert call_options(run("run_all.sh", "--variants", "ops-shmem-base").stdout)[0]["--profile"] == "2"
    cmd, = call_options(run("run_all.sh", "--variants", "base,ops-shmem-base", "--profile", "0").stdout)
    assert cmd["--variants"] == "base,ops-shmem-base" and cmd["--profile"] == "0"
