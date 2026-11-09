#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
cann=${ASCEND_HOME_PATH:-}
shmem=${SHMEM_ROOT:-}
build="$root/build/ll-cpp-base"
data_l2=0 ready=0 notify=0 coalesce=0 coalesce_rows=0
batch_final=OFF local_first=OFF poll_nop=0 ranks_per_host=16 issue=OFF probe=OFF simt_meta=OFF simt_flag=OFF prefetch_return=OFF
small=OFF early=OFF profile=OFF remote_first=OFF defer_return=OFF recv=0 jobs=4 variant=auto ops_source=${OPS_SOURCE_DIR:-}
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'build.sh --shmem-root SDK [--cann-home SDK] [--mpi-home DIR] [--build-dir DIR]'
            echo '         [--ready 0..33] (4 snapshot; 9 WQE preparation; 10 MTE packing; 11 WQE + input prefetch; 12 input6; 13 input-ready; 14 input-pair; 15 input-pipe; 16 input-stream; 17 input-front; 18 input-interleave; 19 input-l2; 20 input6-l2; 21 input6-rpipe; 22 input6-rfull; 23 input6-l2-pipe; 24 setup; 25 single; 26 lean; 27 route; 28 route-only; 29 route-plan; 30 route-priority; 31 tiny-setup; 32 tiny-probe; 33 tiny)'
            echo '         [--data-l2 0..7] stage=1, RDMA TX=2, result/flags=4; requires ready=20 or 23..33'
            echo '         [--notify 0|1|2] (MTE rank / SIMT rank / SIMT per-core)'
            echo '         [--coalesce 0|1|2] [--coalesce-rows 0|1] (off/card/server; whole bucket/one hidden)'
            echo '         [--variant base|opt|ops-shmem-base|ops-shmem-retire|ops-shmem-overlap|ops-shmem-cq|ops-shmem-batch|auto] [--profile ON|OFF] [--small-batch ON|OFF] [--early-reduce ON|OFF] [--recv-cores 0|8..36] [--jobs N]'
            echo '         [--ops-source DIR] optional pinned ops-transformer checkout'
            echo '         [--remote-first ON|OFF] [--defer-return ON|OFF] [--early-issue ON|OFF] [--progress-probe ON|OFF] [--ranks-per-host 4|16]'
            echo '         [--local-first ON|OFF] [--poll-nop 0..65535] [--simt-meta ON|OFF] [--simt-flag ON|OFF] [--prefetch-return ON|OFF]'
            echo '         [--batch-final ON|OFF] requires --local-first ON; vector batch checks and pipelined reduction'
            exit 0;;
        --data-l2|--ready|--notify|--shmem-root|--cann-home|--mpi-home|--build-dir|--variant|--profile|--small-batch|--early-reduce|--recv-cores|--jobs|--remote-first|--defer-return|--ops-source|--early-issue|--progress-probe|--ranks-per-host|--simt-meta|--simt-flag|--prefetch-return|--poll-nop|--local-first|--batch-final|--coalesce|--coalesce-rows)
            (($# >= 2)) || { echo "Missing value: $1" >&2; exit 2; }
            key=$1 value=$2; shift 2
            case "$key" in
                --shmem-root) shmem=$value;; --cann-home) cann=$value;; --mpi-home) mpi=$value;;
                --build-dir) build=$value;; --small-batch) small=$value;; --early-reduce) early=$value;;
                --variant) variant=$value;; --profile) profile=$value;; --recv-cores) recv=$value;; --jobs) jobs=$value;;
                --remote-first) remote_first=$value;; --defer-return) defer_return=$value;;
                --ops-source) ops_source=$value;; --early-issue) issue=$value;; --progress-probe) probe=$value;; --ranks-per-host) ranks_per_host=$value;;
                --local-first) local_first=$value;; --poll-nop) poll_nop=$value;; --simt-meta) simt_meta=$value;; --simt-flag) simt_flag=$value;; --prefetch-return) prefetch_return=$value;;
                --data-l2) data_l2=$value;;
                --ready) ready=$value;;
                --notify) notify=$value;;
                --batch-final) batch_final=$value;;
                --coalesce) coalesce=$value;; --coalesce-rows) coalesce_rows=$value;;
            esac;;
        *) echo "Unknown build option: $1" >&2; exit 2;;
    esac
done
[[ $data_l2 =~ ^[0-7]$ ]] || { echo 'data-l2 must be in [0,7]' >&2; exit 2; }
[[ $data_l2 == 0 || $ready == 20 || $ready =~ ^(2[3-9]|3[0-3])$ ]] || { echo 'data-l2 requires ready=20 or 23..33' >&2; exit 2; }
[[ $ready =~ ^([0-9]|[12][0-9]|3[0-3])$ ]] || { echo 'ready must be an integer in [0,33]' >&2; exit 2; }
if [[ $ready != 0 && ( $notify != 0 || $coalesce != 0 || $small == ON || $simt_meta == ON ) ]]; then
    echo 'Test ready modes separately from notify, coalescing, small-batch and SIMT metadata' >&2; exit 2
fi
[[ $notify =~ ^[012]$ ]] || { echo 'notify must be 0, 1 or 2' >&2; exit 2; }
if [[ $notify != 0 && $coalesce != 0 ]]; then
    echo 'Test notification and coalescing experiments separately' >&2; exit 2
fi
[[ $coalesce =~ ^[012]$ && $coalesce_rows =~ ^[01]$ ]] || { echo 'Invalid coalesce options' >&2; exit 2; }
if [[ $coalesce != 0 ]]; then
    if [[ $variant == auto ]]; then variant=opt; fi
    [[ $variant == opt ]] || { echo 'Use --variant opt for coalescing' >&2; exit 2; }
elif [[ $coalesce_rows != 0 ]]; then
    echo 'coalesce-rows requires coalesce=1 or 2' >&2; exit 2
fi
[[ $profile =~ ^(ON|OFF)$ && $small =~ ^(ON|OFF)$ && $early =~ ^(ON|OFF)$ && $recv =~ ^(0|[89]|[12][0-9]|3[0-6])$ ]] || { echo 'Invalid Combine build option' >&2; exit 2; }
[[ $variant =~ ^(base|opt|ops-shmem-base|ops-shmem-retire|ops-shmem-overlap|ops-shmem-cq|ops-shmem-batch|auto)$ ]] || { echo 'Unknown variant' >&2; exit 2; }
[[ $remote_first =~ ^(ON|OFF)$ && $defer_return =~ ^(ON|OFF)$ ]] || { echo 'Return options must be ON or OFF' >&2; exit 2; }
[[ $ranks_per_host =~ ^(4|16)$ && $issue =~ ^(ON|OFF)$ && $probe =~ ^(ON|OFF)$ ]] || { echo 'Invalid topology or experiment option' >&2; exit 2; }
[[ $simt_meta =~ ^(ON|OFF)$ && $simt_flag =~ ^(ON|OFF)$ && $prefetch_return =~ ^(ON|OFF)$ ]] || { echo 'SIMT/prefetch options must be ON or OFF' >&2; exit 2; }
[[ $local_first =~ ^(ON|OFF)$ ]] || { echo 'local-first must be ON or OFF' >&2; exit 2; }
[[ $batch_final =~ ^(ON|OFF)$ ]] || { echo 'batch-final must be ON or OFF' >&2; exit 2; }
if [[ $batch_final == ON && $local_first != ON ]]; then
    echo 'batch-final requires --local-first ON' >&2; exit 2
fi
if [[ $local_first == ON && ( $early == ON || $probe == ON || $simt_flag == ON ) ]]; then
    echo 'local-first cannot combine with early-reduce, progress-probe or simt-flag' >&2; exit 2
fi
[[ $poll_nop =~ ^(0|[1-9][0-9]{0,4})$ ]] && ((poll_nop<=65535)) || { echo 'poll-nop must be in [0,65535]' >&2; exit 2; }
if [[ $simt_flag == ON && ( $probe == ON || $early == ON ) ]]; then
    echo 'SIMT flag checks cannot be combined with progress-probe or early-reduce' >&2; exit 2
fi
if [[ $prefetch_return == ON && $issue == ON ]]; then
    echo 'prefetch-return and early-issue select different submission points' >&2; exit 2
fi
if [[ $variant == auto ]]; then
    variant=base
    if [[ $profile == ON || $small == ON || $early == ON || $recv != 0 || $remote_first == ON || $defer_return == ON || $issue == ON || $probe == ON || $simt_meta == ON || $simt_flag == ON || $prefetch_return == ON || $local_first == ON || $poll_nop != 0 || $notify != 0 || $ready != 0 ]]; then variant=opt; fi
fi
original=OFF ops_base=OFF ops_policy=0
if [[ $variant == ops-shmem-* ]]; then
    [[ $small == OFF && $early == OFF && $recv == 0 && $remote_first == OFF && $defer_return == OFF && $issue == OFF && $probe == OFF && $simt_meta == OFF && $simt_flag == OFF && $prefetch_return == OFF && $local_first == OFF && $poll_nop == 0 && $notify == 0 && $ready == 0 ]] || { echo 'ops-shmem-base is an independent kernel' >&2; exit 2; }
    ops_base=ON
    [[ $variant != ops-shmem-retire ]] || ops_policy=1
    [[ $variant != ops-shmem-overlap ]] || ops_policy=2
    [[ $variant != ops-shmem-cq ]] || ops_policy=3
    [[ $variant != ops-shmem-batch ]] || ops_policy=4
    if [[ $build == "$root/build/ll-cpp-base" ]]; then build="$root/build/ll-cpp-$variant"; fi
fi
if [[ $variant == base ]]; then
    [[ $profile == OFF && $small == OFF && $early == OFF && $recv == 0 && $remote_first == OFF && $defer_return == OFF && $issue == OFF && $probe == OFF && $simt_meta == OFF && $simt_flag == OFF && $prefetch_return == OFF && $local_first == OFF && $poll_nop == 0 && $notify == 0 && $ready == 0 ]] || { echo 'Use --variant opt for experimental options' >&2; exit 2; }
    original=ON
fi
[[ $jobs =~ ^[0-9]+$ ]] && ((jobs>=1 && jobs<=64)) || { echo 'jobs must be in [1,64]' >&2; exit 2; }
[[ -x "$mpi/bin/mpicxx" && -x "$cann/tools/bisheng_compiler/bin/bisheng" ]] || { echo 'Load CANN and select the MPI installation first' >&2; exit 2; }
[[ -f "$shmem/deepep-sdk.json" && -f "$shmem/lib/aclshmem_bootstrap_mpi.so" ]] || { echo 'Select the identified HNS_1825 SHMEM SDK with its MPI bootstrap plugin' >&2; exit 2; }
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$mpi/lib64:$shmem/lib:${LD_LIBRARY_PATH:-}"
cmake "-DPROJECT_ROOT=$root" "-DSDK=$shmem" -DMODE=check \
    -P "$root/tests/ll_cpp_bench/sdk_identity.cmake"
cmake -S "$root/tests/ll_cpp_bench" -B "$build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release "-DMPI_CXX_COMPILER=$mpi/bin/mpicxx" \
    -DDEEPEP_LL_PYTHON_BENCH=OFF \
    "-DASCEND_HOME_PATH=$cann" "-DSHMEM_ROOT=$shmem" \
    "-DDEEPEP_LL_OPS_BASE=$ops_base" "-DLL_OPS_SOURCE_DIR=$ops_source" \
    "-DDEEPEP_LL_OPS_POLICY=$ops_policy" \
    "-DDEEPEP_LL_COMBINE_ORIGINAL=$original" "-DDEEPEP_LL_COMBINE_PROFILE=$profile" "-DDEEPEP_DEVICE_JOBS=$jobs" "-DDEEPEP_LL_COMBINE_SMALL_BATCH=$small" \
    "-DDEEPEP_LL_COMBINE_RECV_CORES=$recv" "-DDEEPEP_LL_COMBINE_EARLY_REDUCE=$early" \
    "-DDEEPEP_LL_COMBINE_REMOTE_FIRST=$remote_first" "-DDEEPEP_LL_COMBINE_DEFER_RETURN=$defer_return" \
    "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=$ranks_per_host" \
    "-DDEEPEP_LL_COMBINE_EARLY_ISSUE=$issue" "-DDEEPEP_LL_COMBINE_PROGRESS_PROBE=$probe" \
    "-DDEEPEP_LL_COMBINE_SIMT_META=$simt_meta" "-DDEEPEP_LL_COMBINE_SIMT_FLAG=$simt_flag" \
    "-DDEEPEP_LL_COMBINE_PREFETCH_RETURN=$prefetch_return" \
    "-DDEEPEP_LL_COMBINE_LOCAL_FIRST=$local_first" \
    "-DDEEPEP_LL_COMBINE_BATCH_FINAL=$batch_final" \
    "-DDEEPEP_LL_COMBINE_DATA_L2=$data_l2" \
    "-DDEEPEP_LL_COMBINE_READY=$ready" \
    "-DDEEPEP_LL_COMBINE_NOTIFY=$notify" \
    "-DDEEPEP_LL_COMBINE_COALESCE=$coalesce" "-DDEEPEP_LL_COMBINE_COALESCE_ROWS=$coalesce_rows" \
    "-DDEEPEP_LL_COMBINE_POLL_NOP=$poll_nop"
cmake --build "$build" --parallel "$jobs"
printf 'Built runner: %s/ll_combine_bench\nVariant module: %s/libll_bench_variant.so\n' "$build" "$build"
