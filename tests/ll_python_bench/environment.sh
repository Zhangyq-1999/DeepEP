#!/usr/bin/env bash
# Sourced after options are parsed; preserve the selected Python and SDK paths.
ll_python_environment() {
    local env_name=$1 conda_bin
    if [[ ${CONDA_DEFAULT_ENV:-} != "$env_name" && ${CONDA_PREFIX:-} != "$env_name" ]]; then
        conda_bin=${CONDA_EXE:-}
        if [[ -z $conda_bin ]]; then conda_bin=$(type -P conda || true); fi
        if [[ -z $conda_bin ]]; then
            for conda_bin in "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" /home/miniconda3/bin/conda; do
                [[ -x $conda_bin ]] && break
            done
        fi
        [[ -x $conda_bin ]] || { echo "Activate conda $env_name first, or set CONDA_EXE" >&2; return 2; }
        set +u
        eval "$("$conda_bin" shell.bash hook)"
        conda activate "$env_name"
        set -u
    fi
    source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
    ll_bench_environment "$cann" "$mpi" "$shmem"
    [[ -n ${CONDA_PREFIX:-} ]] || { echo 'Conda activation did not set CONDA_PREFIX' >&2; return 2; }
    ll_python="$CONDA_PREFIX/bin/python"
    [[ -x $ll_python ]] || { echo "Missing Conda Python: $ll_python" >&2; return 2; }
    unset PYTHONHOME PYTHONPATH
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
    export PATH="$CONDA_PREFIX/bin:$PATH"
    "$ll_python" - "$env_name" <<'PY'
import os, sys
from pathlib import Path
prefix = Path(os.environ['CONDA_PREFIX']).resolve()
requested = sys.argv[1]
expected = Path(requested).resolve() if '/' in requested else None
if Path(sys.prefix).resolve() != prefix or (expected != prefix if expected else prefix.name != requested):
    raise SystemExit(f'Wrong Python environment: requested={requested}, prefix={sys.prefix}')
print(f'ENV={prefix.name} PYTHON={sys.executable}', flush=True)
PY
}
