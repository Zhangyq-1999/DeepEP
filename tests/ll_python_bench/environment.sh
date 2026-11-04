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
    export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
}
