"""CPU admission tests for SDK identity, independent of CANN."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys
import pytest
from scripts.shmem_sdk_identity import compiled_hns_1825_pi_on_chip, recorded_hns_1825_pi_on_chip, validate_build_identity
from scripts.record_shmem_sdk import check, record

BASE = {"cann_version": "9.3.0", "cann_timestamp": "test", "target": "Ascend950", "source_patches": []}

@pytest.mark.parametrize("rdma", [False, True])
@pytest.mark.parametrize("modified", [False, True])
def test_known_capability_and_explicit_source_changes(rdma, modified):
    identity = deepcopy(BASE)
    if rdma:
        identity["rdma_backend"] = "HNS_1825"
        identity["hns_1825_pi_on_chip"] = 0
    if modified:
        identity["source_patches"] = [{"kind": "local-worktree", "sha256": "a" * 64}]
    validate_build_identity(identity, require_rdma=rdma)

@pytest.mark.parametrize("change", ["extra", "missing", "backend", "patches", "empty", "type"])
def test_reject_unknown_or_incomplete_identity(change):
    identity = deepcopy(BASE)
    if change == "extra": identity["obsolete_extension"] = "value"
    if change == "missing": del identity["cann_timestamp"]
    if change == "backend": identity["rdma_backend"] = "unknown"
    if change == "patches": identity["source_patches"] = "unknown"
    if change == "empty": identity["cann_version"] = ""
    if change == "type": identity = []
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(identity)

def test_rdma_requires_attested_backend():
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(BASE, require_rdma=True)

def test_record_without_build_attestation_does_not_write(tmp_path):
    with pytest.raises(RuntimeError, match="identity"):
        record(tmp_path, "unverified")
    assert not (tmp_path / "deepep-sdk.json").exists()


@pytest.mark.parametrize("mode", [0, 1])
def test_hns_identity_records_both_firmware_modes(mode):
    validate_build_identity({**BASE, "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}, require_rdma=True)


@pytest.mark.parametrize("mode", [None, -1, 2, "0", "1", True, False, 0.0])
def test_hns_identity_rejects_missing_or_ambiguous_mode(mode):
    identity = {**BASE, "rdma_backend": "HNS_1825"}
    if mode is not None:
        identity["hns_1825_pi_on_chip"] = mode
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity(identity)


def test_non_rdma_identity_cannot_attest_hns_mode():
    with pytest.raises(RuntimeError, match="identity"):
        validate_build_identity({**BASE, "hns_1825_pi_on_chip": 0})


@pytest.mark.parametrize("generator", ["make", "ninja", "ninja_multi"])
@pytest.mark.parametrize("mode", [0, 1])
def test_sdk_mode_comes_from_generated_device_rules(tmp_path, generator, mode):
    root = tmp_path / "build with spaces"
    if generator == "make":
        path = root / "src/CMakeFiles/aclshmem_device.dir/flags.make"
        content = f"CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP={mode}\nCXX_FLAGS = -O2\n"
    else:
        path = root / ("build.ninja" if generator == "ninja" else "CMakeFiles/impl-Release.ninja")
        content = ("build src/CMakeFiles/aclshmem_device.dir/device.cpp.o: CXX_COMPILER__aclshmem_device_Release source.cpp\n"
                   f"  DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP={mode}\n  FLAGS = -O2\n\n"
                   "build other.o: CXX_COMPILER__other other.cpp\n  DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=9\n")
    path.parent.mkdir(parents=True)
    path.write_text(content)
    # Upstream's auto-selected variable need not be present in CMakeCache.txt.
    (root / "CMakeCache.txt").write_text("# no doorbell mode cache entry\n")
    assert compiled_hns_1825_pi_on_chip(root) == mode


@pytest.mark.parametrize("definition", ["", "-DACLSHMEMI_HNS_1825_PI_ON_CHIP", "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=2",
                                      "-DACLSHMEMI_HNS_1825_PI_ON_CHIP=0 -DACLSHMEMI_HNS_1825_PI_ON_CHIP=1"])
def test_missing_or_conflicting_device_definition_is_rejected(tmp_path, definition):
    path = tmp_path / "src/CMakeFiles/aclshmem_device.dir/flags.make"
    path.parent.mkdir(parents=True)
    path.write_text(f"CXX_DEFINES = {definition}\n")
    with pytest.raises(RuntimeError, match="doorbell mode"):
        compiled_hns_1825_pi_on_chip(tmp_path)


def test_sdk_mode_does_not_use_header_or_cmake_cache_default(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text("ACLSHMEMI_HNS_1825_PI_ON_CHIP:UNINITIALIZED=1\n")
    with pytest.raises(RuntimeError, match="doorbell mode"):
        compiled_hns_1825_pi_on_chip(tmp_path)


def test_sdk_mode_rejects_disagreement_between_compiled_units(tmp_path):
    (tmp_path / "build.ninja").write_text(
        "build src/CMakeFiles/aclshmem_device.dir/a.cpp.o: CXX_COMPILER__device a.cpp\n"
        "  DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=0\n\n"
        "build src/CMakeFiles/aclshmem_device.dir/b.cpp.o: CXX_COMPILER__device b.cpp\n"
        "  DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=1\n")
    with pytest.raises(RuntimeError, match="inconsistent"):
        compiled_hns_1825_pi_on_chip(tmp_path)


@pytest.mark.parametrize("mode", [0, 1])
def test_probe_mode_reader_uses_attested_sdk_identity(tmp_path, mode):
    (tmp_path / "deepep-sdk.json").write_text(json.dumps({"build": {
        **BASE, "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": mode}}))
    assert recorded_hns_1825_pi_on_chip(tmp_path) == mode


@pytest.fixture
def recording_inputs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "dependencies.lock.json").read_text())["shmem"]
    source = Path(os.environ.get("DEEPEP_TEST_SHMEM_SOURCE", root / "third-party/shmem"))
    if not (source / "include").is_dir():
        pytest.skip("Initialize SHMEM or set DEEPEP_TEST_SHMEM_SOURCE")
    sdk = tmp_path / "installed SDK with spaces"
    for relative in lock["header_sha256_lf"]:
        destination = sdk / "include" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "include" / relative, destination)
    (sdk / "lib").mkdir()
    (sdk / "lib/libshmem.so").write_bytes(b"CPU identity fixture, not a CANN library")
    build_root = tmp_path / "SDK build with spaces"
    flags = build_root / "src/CMakeFiles/aclshmem_device.dir/flags.make"
    flags.parent.mkdir(parents=True)
    flags.write_text("CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP=0\n")
    info = {**BASE, "rdma_backend": "HNS_1825", "hns_1825_pi_on_chip": 0}
    return root, sdk, lock["revision"], build_root, flags, info


@pytest.mark.parametrize("mode", [0, 1])
def test_record_hns_identity_requires_matching_compiled_mode(recording_inputs, mode):
    _, sdk, revision, build_root, flags, info = recording_inputs
    flags.write_text(f"CXX_DEFINES = -DACLSHMEMI_HNS_1825_PI_ON_CHIP={mode}\n")
    info["hns_1825_pi_on_chip"] = mode
    manifest = record(sdk, revision, build_info=info, build_root=build_root)
    assert json.loads(manifest.read_text())["build"]["hns_1825_pi_on_chip"] == mode
    assert check(sdk) == manifest


@pytest.mark.parametrize("fault", ["missing_argument", "missing_graph", "mode_mismatch"])
def test_record_missing_or_conflicting_proof_never_writes_manifest(recording_inputs, fault):
    _, sdk, revision, build_root, flags, info = recording_inputs
    if fault == "missing_argument":
        build_root = None
    elif fault == "missing_graph":
        flags.unlink()
    else:
        info["hns_1825_pi_on_chip"] = 1
    with pytest.raises((ValueError, RuntimeError), match="build graph|doorbell mode"):
        record(sdk, revision, build_info=info, build_root=build_root)
    assert not (sdk / "deepep-sdk.json").exists()


def test_record_cli_accepts_build_proof_and_preserves_verification(recording_inputs, tmp_path):
    root, sdk, revision, build_root, _, info = recording_inputs
    info_path = tmp_path / "build info.json"
    info_path.write_text(json.dumps(info))
    script = root / "scripts/record_shmem_sdk.py"
    command = [sys.executable, str(script), "--sdk", str(sdk)]
    subprocess.run([*command, "--revision", revision, "--build-info", str(info_path),
                    "--build-root", str(build_root)], cwd=tmp_path, check=True, capture_output=True, text=True)
    manifest = sdk / "deepep-sdk.json"
    original = manifest.read_bytes()
    for option in ("--check", "--verify"):
        subprocess.run([*command, option], cwd=tmp_path, check=True, capture_output=True, text=True)
        assert manifest.read_bytes() == original
