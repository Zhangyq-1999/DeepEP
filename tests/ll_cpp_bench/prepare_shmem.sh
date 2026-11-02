#!/usr/bin/env bash
# Build the pinned SDK with Shell/CMake. No DeepEP Python package is installed.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
: "${ASCEND_HOME_PATH:?Load the CANN environment first}"
[[ -x "$mpi/bin/mpicxx" ]] || { echo 'MPI compiler unavailable' >&2;exit 2; }
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$mpi/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-4}
src="$root/third-party/shmem"
git -C "$root" submodule update --init --recursive -- third-party/shmem
for part in "$src" "$src/build" "$src/install" "$src/ci" "$src/3rdparty";do
    [[ ! -L "$part" ]] || { echo "Refusing to build through symlink: $part" >&2;exit 2; }
done
cmake "-DPROJECT_ROOT=$root" "-DSHMEM_SOURCE=$src" -DMODE=source \
    -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
sdk="$src/install/shmem"
if [[ -f "$sdk/deepep-sdk.json" ]];then
    cmake "-DPROJECT_ROOT=$root" "-DSDK=$sdk" -DMODE=check \
        -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
else
    [[ ! -e "$src/build" && ! -e "$src/install" ]] || {
        echo 'Unidentified SDK build/install exists. Select an existing SDK or use a fresh checkout.' >&2;exit 2;
    }
    (cd "$src" && bash -e scripts/build.sh -soc_type Ascend950 -enable_rdma -rdma_backend HNS_1825)
    cmake "-DPROJECT_ROOT=$root" "-DSHMEM_SOURCE=$src" -DMODE=source \
        -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
    cmake "-DPROJECT_ROOT=$root" "-DSDK=$sdk" "-DSDK_BUILD=$src/build" \
        "-DASCEND_HOME_PATH=$ASCEND_HOME_PATH" -DMODE=record \
        -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
fi
[[ -f "$sdk/lib/aclshmem_bootstrap_mpi.so" ]] || { echo 'SHMEM MPI plugin was not built' >&2;exit 2; }
printf 'SHMEM_ROOT=%s\n' "$sdk"
