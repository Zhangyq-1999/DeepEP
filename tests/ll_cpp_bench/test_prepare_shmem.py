"""Exercise SDK replacement without allowing a stale build identity to survive."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("fail_build", [False, True])
def test_explicit_rebuild_invalidates_old_sdk(tmp_path, fail_build):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("Bash required")
    root = tmp_path / "repo with spaces"
    scripts = root / "tests/ll_cpp_bench"
    scripts.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("prepare_shmem.sh"), scripts / "prepare_shmem.sh")
    src = root / "third-party/shmem"
    sdk = src / "install/shmem"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "lib/aclshmem_bootstrap_mpi.so").touch()
    manifest = sdk / "deepep-sdk.json"
    manifest.write_text("old identity")
    (src / "scripts").mkdir()
    (src / "scripts/build.sh").write_text('''#!/bin/bash
set -eu
[[ ! -e install/shmem/deepep-sdk.json ]]
touch build-started
[[ ${FAIL_BUILD:-0} == 0 ]] || exit 17
''', newline="\n")
    bindir = tmp_path / "mpi/bin"
    bindir.mkdir(parents=True)
    for name, content in {
        "git": "exit 0\n",
        "mpicxx": "exit 0\n",
        "cmake": '''set -eu
for arg in "$@";do
  case "$arg" in -DSDK=*) sdk=${arg#-DSDK=};; -DMODE=*) mode=${arg#-DMODE=};; esac
done
if [[ $mode == check ]];then exit 29;fi
if [[ $mode == record ]];then printf 'new identity' > "$sdk/deepep-sdk.json";fi
''',
    }.items():
        path = bindir / name
        path.write_text("#!/bin/bash\n" + content, newline="\n")
        path.chmod(0o755)
    mpi_home = bindir.parent.as_posix()
    if os.name == "nt":
        mpi_home = "/" + mpi_home[0].lower() + mpi_home[2:]
    env = dict(os.environ, MPI_HOME=mpi_home, ASCEND_HOME_PATH="test-cann",
               FAIL_BUILD=str(int(fail_build)))
    cmd = [bash, str(scripts / "prepare_shmem.sh")]
    # Normal reuse must reject the old identity, rather than rebuilding silently.
    old = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert old.returncode == 29 and not (src / "build-started").exists(), old.stderr
    result = subprocess.run(cmd + ["--rebuild"], env=env, capture_output=True, text=True)
    assert result.returncode == (17 if fail_build else 0), result.stdout + result.stderr
    assert (src / "build-started").exists()
    assert not manifest.exists() if fail_build else manifest.read_text() == "new identity"
