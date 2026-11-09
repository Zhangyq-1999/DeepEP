"""Exercise convenience scripts with fake builds/launches; no NPU required."""
import os
from pathlib import Path
import shutil
import shlex
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
PROFILES = ('ops-shmem-cq-profile', 'ops-shmem-batch-profile')
CLEAN = ('base', 'input6-l2-route-only', 'ops-shmem-overlap', 'ops-shmem-cq', 'ops-shmem-batch')


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

    def run(script, *args, ok=True, real_launch=False):
        if real_launch:
            shutil.copyfile(HERE / "launch.sh", scripts / "launch.sh")
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
    assert len(calls) == len(CLEAN) + len(PROFILES)
    for name, call in zip((*CLEAN, *PROFILES), calls):
        assert call["--build-dir"].endswith("/build/ll-cpp-" + name)
        assert call["--variant"] == (name.removesuffix("-profile") if name.removesuffix("-profile") in ("base", "ops-shmem-base", "ops-shmem-retire", "ops-shmem-overlap", "ops-shmem-cq", "ops-shmem-batch") else "opt")
        assert call["--profile"] == ("ON" if name.endswith("-profile") else "OFF")
        assert call["--small-batch"] == ("ON" if name == "opt-profile" else "OFF")
        assert call["--recv-cores"] == ("32" if name.startswith(("ready-r32", "input6-r32")) else "36" if name.startswith("ready-r36") else "0")
        assert call["--early-reduce"] == ("ON" if name == "early-profile" else "OFF")
        assert call["--remote-first"] == ("ON" if (name.startswith(("notify-", "input6-l2-remote", "input6-remote", "input6-rpipe", "input6-rfull")) or name in ("remote", "return", "remote-profile", "return-profile")) else "OFF")
        assert call["--defer-return"] == ("ON" if name in ("pipeline", "return", "pipeline-profile", "return-profile", "ready-pipeline", "ready-pipeline-profile") else "OFF")
        assert call["--data-l2"] == str(next((mask for suffix,mask in (("tiny-setup",7),("tiny-probe",7),("tiny",7),("route-priority",7),("route-tx-bypass",5),("stage",1),("tx",2),("rx",4),("all",7),("pipe",7),("setup",7),("single",7),("lean",7),("route",7),("route-only",7),("route-plan",7)) if name.removesuffix("-profile")=="input6-l2-"+suffix),0))
        assert call["--ranks-per-host"] == "4"
        assert call["--early-issue"] == ("ON" if name in ("issue", "issue-probe", "issue-profile", "issue-probe-profile") else "OFF")
        assert call["--progress-probe"] == ("ON" if name in ("probe", "issue-probe", "probe-profile", "issue-probe-profile") else "OFF")
        assert call["--simt-meta"] == ("ON" if name.startswith("prepare") else "OFF")
        assert call["--prefetch-return"] == ("ON" if name.startswith("batch-pipeline") else "OFF")
        assert call["--simt-flag"] == "OFF"
        assert call["--ready"] == ("31" if name.removesuffix("-profile")=="input6-l2-tiny-setup" else "32" if name.removesuffix("-profile")=="input6-l2-tiny-probe" else "33" if name.removesuffix("-profile")=="input6-l2-tiny" else "23" if name.startswith("input6-l2-pipe") else "24" if name.startswith("input6-l2-setup") else "25" if name.startswith("input6-l2-single") else "26" if name.startswith("input6-l2-lean") else "30" if name.startswith("input6-l2-route-priority") else "28" if name.startswith(("input6-l2-route-only", "input6-l2-route-tx-bypass")) else "29" if name.startswith("input6-l2-route-plan") else "27" if name.startswith("input6-l2-route") else "20" if name.startswith("input6-l2") else "19" if name.startswith("input-l2") else "18" if name.startswith("input-interleave") else "16" if name.startswith("input-stream") else "17" if name.startswith("input-front") else "14" if name.startswith("input-pair") else "15" if name.startswith("input-pipe") else "21" if name.startswith("input6-rpipe") else "22" if name.startswith("input6-rfull") else "12" if name.startswith("input6") else "13" if name.startswith("input-ready") else "11" if name.startswith("wqe-prefetch") else "9" if name.startswith("wqe-prep") else "4" if name.startswith("ready-snapshot") else "0")
        assert call["--notify"] == ("1" if name.startswith("notify-simt") else "2" if name.startswith("notify-core") else "0")
        assert call["--local-first"] == ("ON" if name.startswith("batch") else "OFF")
        assert call["--batch-final"] == ("ON" if name.startswith("batch") else "OFF")
        assert call["--poll-nop"] == ("800" if name.startswith("poll800") else "0")
        assert call["--coalesce"] == ("1" if name.startswith("card-") else "2" if name.startswith("server-") else "0")
        assert call["--coalesce-rows"] == ("1" if "-row" in name else "0")
        assert call["--shmem-root"].endswith("/third-party/shmem/install/shmem")
    env["FAIL_BUILD"] = PROFILES[0]
    result = run("build_all.sh", ok=False)
    assert result.returncode == 17 and len(call_options(result.stdout)) == len(CLEAN)+1
    assert "CALL" not in run("build_all.sh", "--variants", "base,typo", ok=False).stdout


def test_source_index_comparison_builds(harness):
    run, _ = harness
    names = ("input6-l2-route-only", "input6-l2-route-i64", "batch", "batch-i64")
    names += tuple(name + "-profile" for name in names)
    calls = call_options(run("build_all.sh", "--variants", ",".join(names)).stdout)
    assert len(calls) == 8
    for name, call in zip(names, calls):
        assert call["--source-index-bits"] == ("64" if "i64" in name else "32")
        assert call["--ready"] == ("0" if name.startswith("batch") else "28")
        assert call["--data-l2"] == ("0" if name.startswith("batch") else "7")
        assert call["--local-first"] == call["--batch-final"] == ("ON" if name.startswith("batch") else "OFF")
        assert call["--profile"] == ("ON" if name.endswith("-profile") else "OFF")


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


def test_optional_fusion_is_explicit_and_old_layouts_remain_available(harness):
    run, _ = harness
    calls = call_options(run("build_all.sh", "--variants",
                            "input6-fused,input6-fused-profile,input-interleave,input-l2").stdout)
    assert [c["--ready"] for c in calls] == ["20", "20", "18", "19"]
    assert [c["--recv-cores"] for c in calls] == ["32", "32", "0", "0"]
    assert [c["--remote-first"] for c in calls] == ["ON", "ON", "OFF", "OFF"]
    assert [c["--profile"] for c in calls] == ["OFF", "ON", "OFF", "OFF"]


def test_small_task_variants_are_independent(harness):
    run, _ = harness
    names = ["input6-l2-" + s for s in ("setup", "single", "lean", "route", "route-only", "route-plan")]
    calls = call_options(run("build_all.sh", "--variants", ",".join(names)).stdout)
    assert [c["--ready"] for c in calls] == ["24", "25", "26", "27", "28", "29"]
    assert all(c["--data-l2"] == "7" and c["--small-batch"] == "OFF" and
               c["--recv-cores"] == "0" and c["--remote-first"] == "OFF" for c in calls)


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
        assert call["--variants"] == ",".join((*CLEAN, *PROFILES))
        assert call["--tokens"] == "1,2,4,8,16,24"
        assert call["--iters"] == "100" and call["--profile"] == "2"
        assert call["--profile-iters"] == "10"
        assert call["--profile-frequency-hz"] == "1000000000"
        assert call["--entry-sync"] == "device"
        assert f"run_all.sh {ids[-1]} \n" in result.stdout
    assert len(set(ids)) == 2
    env["TEST_LOCAL_IP"] = "90.90.93.184"
    joined, = call_options(run("run_all.sh", ids[0]).stdout)
    assert joined["--node-rank"] == "1" and joined["--run-id"] == ids[0]
    assert "Copy RUN_ID" in run("run_all.sh", ok=False).stderr


@pytest.mark.parametrize("nodes", [2, 4])
def test_default_suite_reaches_real_launcher_on_all_hosts(harness, nodes):
    run, env = harness
    run_id = "ready-suite-launch-regression"
    hosts = ("90.90.93.84", "90.90.93.184", "90.90.93.85", "90.90.93.185")[:nodes]
    options = [] if nodes == 2 else ["--ep", "64", "--ranks-per-host", "16", "--hosts", ",".join(hosts)]
    for node, host in enumerate(hosts):
        env["TEST_LOCAL_IP"] = host
        result = run("run_all.sh", run_id, *options, "--dry-run", real_launch=True)
        for phase, name in enumerate((*CLEAN, *PROFILES)):
            assert f"phase={phase} variant={name} output=" in result.stdout
        assert f"--proxy-id {node}" in result.stdout
        assert "one shared initialization" in result.stdout


def test_host_detection_and_benchmark_overrides(harness):
    run, env = harness
    env["TEST_LOCAL_IP"] = "unlisted-host"
    assert "--node-rank" in run("run_all.sh", ok=False).stderr
    result = run("run_all.sh", "--node-rank", "0", "--hosts", "a,b", "--master-port", "30071",
                 "--variants", "return-profile,ref-profile", "--tokens", "72,256", "--iters", "20",
                 "--", "--warmup", "5", "--seed", "4", "--route", "normal")
    call, = call_options(result.stdout)
    assert call["--tokens"] == "72,256" and call["--iters"] == "20" and call["--warmup"] == "5"
    assert call["--master-addr"] == "a" and call["--master-port"] == "30071"
    assert call["--route"] == "normal" and call["--seed"] == "4"
    assert "route=normal seed=4" in result.stdout
    peer_line = next(line for line in result.stdout.splitlines() if line.startswith("Run on node 1:"))
    peer_args = shlex.split(peer_line.removeprefix("Run on node 1: "))
    assert peer_args[peer_args.index("--hosts") + 1] == "a,b"
    assert peer_args[peer_args.index("--variants") + 1] == "return-profile,ref-profile"
    assert peer_args[peer_args.index("--node-rank") + 1] == "1"
    assert peer_args[peer_args.index("--master-port") + 1] == "30071"
    local, = call_options(run("run_all.sh", "--ep", "4", "--devices", "6,7,14,15").stdout)
    assert local["--hosts"] == "localhost" and local["--node-rank"] == "0"
    assert local["--devices"] == "6,7,14,15" and local["--entry-sync"] == "device"
    host, = call_options(run("run_all.sh", "--node-rank", "0", "--entry-sync", "host").stdout)
    assert host["--entry-sync"] == "host"
    assert call_options(run("run_all.sh", "--ep", "16", "--ranks-per-host", "16", "--variants", "base").stdout)[0]["--profile"] == "2"
    clean, = call_options(run("run_all.sh", "--ep", "16", "--ranks-per-host", "16", "--variants", "base", "--profile", "0").stdout)
    assert clean["--profile"] == "0"
    env["TEST_LOCAL_IP"] = "90.90.93.84 90.90.93.184"
    assert "Multiple hosts match" in run("run_all.sh", ok=False).stderr


def test_four_host_commands_and_detection(harness):
    run, env = harness
    builds = call_options(run("build_all.sh", "--ranks-per-host", "16").stdout)
    assert len(builds) == len(CLEAN)+len(PROFILES) and all(call["--ranks-per-host"] == "16" for call in builds)
    assert [call["--ready"] for call in builds] == ["0", "28", "0", "0", "0", "0", "0"]
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


def test_custom_mpi_environment_travels_in_peer_commands(harness):
    run, env = harness
    env["MPI_HOME"] = "/home/user/mpi 4.1.3"
    result = run("run_all.sh", "--ep", "64", "--ranks-per-host", "16", "--hosts", "a,b,c,d", "--node-rank", "0")
    lines = [line for line in result.stdout.splitlines() if line.startswith("Run on node ")]
    assert len(lines) == 3
    del env["MPI_HOME"]  # Other terminals do not inherit node 0's export.
    for line in lines:
        command = shlex.split(line.split(": ", 1)[1])
        peer, = call_options(run("run_all.sh", *command[2:]).stdout)
        assert peer["--mpi-home"] == "/home/user/mpi 4.1.3"


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


def test_receive_core_experiments_keep_snapshot_algorithm(harness):
    run, _ = harness
    names = ("ready-r32", "ready-r36", "ready-r32-profile", "ready-r36-profile")
    calls = call_options(run("build_all.sh", "--variants", ",".join(names), "--ranks-per-host", "16").stdout)
    for name, call in zip(names, calls):
        assert call["--ready"] == "4"
        assert call["--recv-cores"] == ("32" if "r32" in name else "36")
        assert call["--ranks-per-host"] == "16"
        assert call["--profile"] == ("ON" if name.endswith("-profile") else "OFF")
        assert call["--small-batch"] == call["--early-reduce"] == "OFF"
