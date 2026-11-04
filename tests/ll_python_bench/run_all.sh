#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
env_name=py311_zyq cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.T101}
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1} shmem=${SHMEM_ROOT:-$root/third-party/shmem/install/shmem}
args=()
while (($#)); do
    case "$1" in
        --conda-env|--cann-home|--mpi-home|--shmem-root)
            (($#>=2)) || { echo "Missing value: $1" >&2; exit 2; }
            case "$1" in --conda-env) env_name=$2;; --cann-home) cann=$2;; --mpi-home) mpi=$2;; --shmem-root) shmem=$2;; esac
            shift 2;;
        *) args+=("$1"); shift;;
    esac
done
source "$root/tests/ll_python_bench/environment.sh"
ll_python_environment "$env_name"
export LL_PY_CONDA_ENV="$env_name"
exec "$ll_python" "$root/tests/ll_python_bench/launch.py" "${args[@]}"
