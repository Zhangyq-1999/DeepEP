#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
variant=base env_name=py311_zyq jobs=4 ranks=4
cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.T101}
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1} shmem=${SHMEM_ROOT:-$root/third-party/shmem/install/shmem}
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'build.sh [--variant base|ref-profile|input6|input6-l2|input6-remote|input6-l2-remote|NAME-profile]'
            echo '         [--conda-env py311_zyq]'
            echo '         [--ranks-per-host 4|16] [--jobs N] [--cann-home DIR] [--shmem-root DIR] [--mpi-home DIR]'
            exit 0;;
        --variant|--conda-env|--ranks-per-host|--jobs|--cann-home|--shmem-root|--mpi-home)
            (($#>=2)) || { echo "Missing value: $1" >&2; exit 2; }
            case "$1" in
                --variant) variant=$2;; --conda-env) env_name=$2;; --ranks-per-host) ranks=$2;;
                --jobs) jobs=$2;; --cann-home) cann=$2;; --shmem-root) shmem=$2;; --mpi-home) mpi=$2;;
            esac; shift 2;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
clean=${variant%-profile}
[[ $variant != ref-profile ]] || clean=base
[[ $variant != base-profile && $variant != ref ]] || { echo 'Use base or ref-profile' >&2; exit 2; }
case "$clean" in base|ready-snapshot|wqe-prep|wqe-prefetch|input6|input-ready|input-pair|input-pipe|input-stream|input-front|input-interleave|input-l2|input6-l2|input6-l2-stage|input6-l2-tx|input6-l2-rx|input6-l2-all|input6-l2-pipe|input6-l2-remote|input6-r32|input6-remote|input6-fused|input6-rpipe|input6-rfull|mte-pack|card-all|card-row|server-all|server-row) ;; *) echo 'Choose a supported clean or -profile variant' >&2; exit 2;; esac
source "$root/tests/ll_python_bench/environment.sh"
ll_python_environment "$env_name"
"$ll_python" -c 'import torch, torch_npu; print("torch:",torch.__version__,"torch_npu:",torch_npu.__version__)'
# Install build tools only into the selected conda environment; keep its Torch/CANN pairing.
if ! "$ll_python" -c 'import pybind11' >/dev/null 2>&1; then "$ll_python" -m pip install 'pybind11>=2.12,<3'; fi
if ! command -v cmake >/dev/null; then "$ll_python" -m pip install 'cmake>=3.20,<4'; fi
bash "$root/tests/ll_cpp_bench/build_all.sh" --variants "$variant" --ranks-per-host "$ranks" \
    --jobs "$jobs" --cann-home "$cann" --mpi-home "$mpi" --shmem-root "$shmem"
build="$root/build/ll-cpp-$variant"
cmake -S "$root/tests/ll_cpp_bench" -B "$build" -DDEEPEP_LL_PYTHON_BENCH=ON \
    "-DDEEPEP_LL_PYTHON_VARIANT=$variant" \
    "-DPython_EXECUTABLE=$ll_python" \
    "-Dpybind11_DIR=$("$ll_python" -m pybind11 --cmakedir)"
cmake --build "$build" --target _ll_py_combine --parallel "$jobs"
cmake --install "$build" --component ll_python
"$ll_python" - "$variant" <<'PY'
import sys, sysconfig
from pathlib import Path
location = Path(sysconfig.get_path('platlib')) / 'deepep_ll_bench' / sys.argv[1]
sys.path.insert(0, str(location))
import _ll_py_combine as m
if Path(m.__file__).resolve().parent != location.resolve():
    raise SystemExit('Loaded an unexpected benchmark extension')
print(f'INSTALLED={m.__file__} ranks_per_host={m.ranks_per_host}', flush=True)
PY
