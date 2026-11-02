#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
cann=${ASCEND_HOME_PATH:-}
shmem=${SHMEM_ROOT:-}
build="$root/build/ll-cpp-base"
small=OFF early=OFF recv=0 jobs=4
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'build.sh --shmem-root SDK [--cann-home SDK] [--mpi-home DIR] [--build-dir DIR]'
            echo '         [--small-batch ON|OFF] [--early-reduce ON|OFF] [--recv-cores 0|8..36] [--jobs N]'
            exit 0;;
        --shmem-root|--cann-home|--mpi-home|--build-dir|--small-batch|--early-reduce|--recv-cores|--jobs)
            (($# >= 2)) || { echo "Missing value: $1" >&2; exit 2; }
            key=$1 value=$2; shift 2
            case "$key" in
                --shmem-root) shmem=$value;; --cann-home) cann=$value;; --mpi-home) mpi=$value;;
                --build-dir) build=$value;; --small-batch) small=$value;; --early-reduce) early=$value;;
                --recv-cores) recv=$value;; --jobs) jobs=$value;;
            esac;;
        *) echo "Unknown build option: $1" >&2; exit 2;;
    esac
done
[[ $small =~ ^(ON|OFF)$ && $early =~ ^(ON|OFF)$ && $recv =~ ^(0|[89]|[12][0-9]|3[0-6])$ ]] || { echo 'Invalid Combine build option' >&2; exit 2; }
[[ $jobs =~ ^[0-9]+$ ]] && ((jobs>=1 && jobs<=64)) || { echo 'jobs must be in [1,64]' >&2; exit 2; }
[[ -x "$mpi/bin/mpicxx" && -x "$cann/tools/bisheng_compiler/bin/bisheng" ]] || { echo 'Load CANN and select the MPI installation first' >&2; exit 2; }
[[ -f "$shmem/deepep-sdk.json" && -f "$shmem/lib/aclshmem_bootstrap_mpi.so" ]] || { echo 'Select the identified HNS_1825 SHMEM SDK with its MPI bootstrap plugin' >&2; exit 2; }
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$mpi/lib64:$shmem/lib:${LD_LIBRARY_PATH:-}"
cmake "-DPROJECT_ROOT=$root" "-DSDK=$shmem" -DMODE=check \
    -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
cmake -S "$root/tests/ll_cpp_bench" -B "$build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release "-DMPI_CXX_COMPILER=$mpi/bin/mpicxx" \
    "-DASCEND_HOME_PATH=$cann" "-DSHMEM_ROOT=$shmem" \
    "-DDEEPEP_DEVICE_JOBS=$jobs" "-DDEEPEP_LL_COMBINE_SMALL_BATCH=$small" \
    "-DDEEPEP_LL_COMBINE_RECV_CORES=$recv" "-DDEEPEP_LL_COMBINE_EARLY_REDUCE=$early"
cmake --build "$build" --parallel "$jobs"
printf 'Built: %s/ll_combine_bench\n' "$build"
