#!/usr/bin/env bash
# Build the frozen original and the isolated diagnostic experiments.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
variants=$ll_bench_variants ranks_per_host=4
cann=$ll_bench_cann mpi=$ll_bench_mpi shmem=$ll_bench_shmem jobs=4 dry=0
fail() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
build_all.sh [--variants NAME[,NAME...]] [--ranks-per-host 4|16] [--jobs N] [--dry-run]
             [--cann-home DIR] [--mpi-home DIR] [--shmem-root DIR]
Defaults: base (clean), followed by ref-profile (diagnostic); no polling pause.
Optional poll800/poll800-profile: Nop<800>() after each failed flag probe.
SIMT selection/check experiments prepare, flag, prepare-flag are opt-in only.
meta/meta-profile selects metadata only; prefetch/prefetch-profile selects overlap only.
--ranks-per-host 4 (default): cross-host EP8, or single-host EP2/4.
--ranks-per-host 16: single-host EP4/8/16, or cross-host EP32.
Previous experiments remote, pipeline, return, opt-profile, r36-profile and early-profile remain
available through --variants but are excluded from the default build.
ops-shmem-base builds the optional ops-transformer baseline with SHMEM transport.
Existing build directories are updated in place; all flags are explicit.
Dispatch compiler outputs are shared; each variant compiles only its own Combine.
SDK defaults: CANN cann-9.2.T101, MPI /usr/local/mpich-3.2.1,
             SHMEM third-party/shmem/install/shmem in this checkout.
ASCEND_HOME_PATH, MPI_HOME and SHMEM_ROOT override these SDK defaults.
USAGE
            exit 0;;
        --dry-run) dry=1; shift;;
        --variants|--jobs|--cann-home|--mpi-home|--shmem-root|--ranks-per-host)
            (($#>=2)) || fail "Missing value: $1"
            key=$1 value=$2; shift 2
            case "$key" in
                --variants) variants=$value;; --jobs) jobs=$value;; --cann-home) cann=$value;;
                --mpi-home) mpi=$value;; --shmem-root) shmem=$value;; --ranks-per-host) ranks_per_host=$value;;
            esac;;
        *) fail "Unknown build option: $1";;
    esac
done
[[ $ranks_per_host =~ ^(4|16)$ ]] || fail 'ranks-per-host must be 4 or 16'
[[ $jobs =~ ^[1-9][0-9]*$ ]] && ((jobs<=64)) || fail 'jobs must be in [1,64]'
[[ $variants =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || fail 'Provide comma-separated experiment names'
IFS=, read -r -a names <<< "$variants"
# Validate the complete selection before starting any build.
for name in "${names[@]}"; do
    case "$name" in
        poll800|poll800-profile|meta|meta-profile|prefetch|prefetch-profile|prepare|flag|prepare-flag|prepare-profile|flag-profile|prepare-flag-profile|issue|probe|issue-probe|issue-profile|probe-profile|issue-probe-profile|base|remote|pipeline|return|ops-shmem-base|ref-profile|opt-profile|r36-profile|early-profile|remote-profile|pipeline-profile|return-profile) ;;
        *) fail "Unknown experiment: $name (see --help)";;
    esac
done
if ((!dry)); then ll_bench_environment "$cann" "$mpi" "$shmem"; fi
for name in "${names[@]}"; do
    variant=opt profile=OFF small=OFF early=OFF remote=OFF defer=OFF issue=OFF probe=OFF recv=0 simt_meta=OFF simt_flag=OFF prefetch_return=OFF poll_nop=0
    if [[ $name == *-profile ]]; then profile=ON; fi
    case "$name" in
        poll800|poll800-profile) poll_nop=800;;
        meta|meta-profile) simt_meta=ON;;
        prefetch|prefetch-profile) prefetch_return=ON;;
        prepare|prepare-profile) simt_meta=ON; prefetch_return=ON;;
        flag|flag-profile) simt_flag=ON;;
        prepare-flag|prepare-flag-profile) simt_meta=ON; simt_flag=ON; prefetch_return=ON;;
        base) variant=base; profile=OFF;;
        ops-shmem-base) variant=ops-shmem-base; profile=OFF;;
        opt-profile) small=ON;;
        r36-profile) recv=36;;
        early-profile) early=ON;;
        remote|remote-profile) remote=ON;;
        pipeline|pipeline-profile) defer=ON;;
        return|return-profile) remote=ON; defer=ON;;
        issue|issue-profile) issue=ON;;
        probe|probe-profile) probe=ON;;
        issue-probe|issue-probe-profile) issue=ON; probe=ON;;
    esac
    cmd=(bash "$root/tests/ll_cpp_bench/build.sh"
         --cann-home "$cann" --mpi-home "$mpi" --shmem-root "$shmem"
         --build-dir "$root/build/ll-cpp-$name" --jobs "$jobs"
         --variant "$variant" --profile "$profile" --small-batch "$small"
         --recv-cores "$recv" --early-reduce "$early" --remote-first "$remote" --defer-return "$defer"
         --early-issue "$issue" --progress-probe "$probe" --simt-meta "$simt_meta" --simt-flag "$simt_flag"
         --prefetch-return "$prefetch_return" --poll-nop "$poll_nop" --ranks-per-host "$ranks_per_host")
    printf '\nBuilding %s\n' "$name"
    if ((dry)); then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
printf '\nBuild selection complete: %s\n' "$variants"
