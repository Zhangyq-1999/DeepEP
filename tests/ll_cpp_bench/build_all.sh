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
Defaults: base/input6/input-stream/input-front/input-pipe, plus matching diagnostics.
Target BS32/48/64; BS72 is a neighboring case and BS256 checks regressions.
base is the frozen original; ready-snapshot remains opt-in.
wqe-prep prepares a private UB WQE while M4 payload staging runs, then publishes it.
wqe-prefetch overlaps the next ready task's input reads with the previous staged return.
input6: wqe-prep with up to six M4 input lanes (previously three).
input-stream: input6 plus alternating FP32 buffers and shared Vector barriers.
input-front: input6 with the anchor product issued before other input reads.
input-pipe: input6 plus WQE/input submission overlap.
input-pair and wqe-prefetch remain opt-in experiments.
input-ready (opt-in): input6 plus per-contribution ready prefetch.
mte-pack (opt-in) writes M4 payload/flags directly to GM with strided MTE.
Both retain snapshot's 28/28 core split; ready-r32/ready-r36 remain opt-in.
ready-fast skips unchanged empty selections and reuses resident metadata caches.
ready-pipeline adds M4-only staging overlap with the next already-ready task.
ready-snapshot loads routing once in the idle payload arena before M4 starts.
stage-prefetch posts after next input reads; stage-pair shares one wait for two M4 copies.
stage-stream sends each ready batch's first packet immediately, then overlaps full packet preparation.
plan-compact adds stable Vector FIRST_HOP compaction to ready-snapshot before server filtering.
stage-stream, plan-compact and other previously measured experiments remain opt-in.
remote: cross-server returns first. notify-simt: remote plus SIMT rank notification.
notify-core: remote plus direct producer-core notification; Vector readiness checks.
card: coalesce returns to the same destination rank; server: relay via the same local slot.
all: one whole bucket WQE; row: one hidden per WQE (7776 B at H3584/BF16).
batch: vector batch readiness, route filtering and pipelined local-first reduction.
batch-pipeline: batch plus next-input prefetch overlapping return staging.
Optional poll800/poll800-profile: Nop<800>() after each failed flag probe.
SIMT selection/check experiments prepare, flag, prepare-flag are opt-in only.
meta/meta-profile selects metadata only; prefetch/prefetch-profile selects overlap only.
--ranks-per-host 4 (default): cross-host EP8, or single-host EP2/4.
--ranks-per-host 16: single-host EP4/8/16, two-host EP32 or four-host EP64.
Previous experiments pipeline, return, opt-profile, r36-profile and early-profile remain
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
        input-stream|input-stream-profile|input-front|input-front-profile|input-pair|input-pair-profile|input-pipe|input-pipe-profile|input6|input6-profile|input-ready|input-ready-profile|wqe-prep|wqe-prep-profile|wqe-prefetch|wqe-prefetch-profile|mte-pack|mte-pack-profile|plan-compact|plan-compact-profile|ready-r32|ready-r32-profile|ready-r36|ready-r36-profile) ;;
        stage-stream|stage-stream-profile|stage-prefetch|stage-prefetch-profile|stage-pair|stage-pair-profile|ready-snapshot|ready-snapshot-profile|ready-fast|ready-pipeline|ready-fast-profile|ready-pipeline-profile|ready-task|ready-shared|ready-task-profile|ready-shared-profile|notify-simt|notify-core|notify-simt-profile|notify-core-profile|card-all|card-row|server-all|server-row|card-all-profile|card-row-profile|server-all-profile|server-row-profile|batch|batch-profile|batch-pipeline|batch-pipeline-profile|local-first|local-first-profile|poll800|poll800-profile|meta|meta-profile|prefetch|prefetch-profile|prepare|flag|prepare-flag|prepare-profile|flag-profile|prepare-flag-profile|issue|probe|issue-probe|issue-profile|probe-profile|issue-probe-profile|base|remote|pipeline|return|ops-shmem-base|ref-profile|opt-profile|r36-profile|early-profile|remote-profile|pipeline-profile|return-profile) ;;
        *) fail "Unknown experiment: $name (see --help)";;
    esac
done
if ((!dry)); then ll_bench_environment "$cann" "$mpi" "$shmem"; fi
for name in "${names[@]}"; do
    ready=0 notify=0 coalesce=0 coalesce_rows=0
    variant=opt profile=OFF small=OFF early=OFF remote=OFF defer=OFF issue=OFF probe=OFF recv=0 simt_meta=OFF simt_flag=OFF prefetch_return=OFF poll_nop=0 local_first=OFF batch_final=OFF
    if [[ $name == *-profile ]]; then profile=ON; fi
    case "$name" in
        ready-task|ready-task-profile) ready=1;;
        ready-fast|ready-fast-profile) ready=3;;
        ready-snapshot|ready-snapshot-profile) ready=4;;
        stage-prefetch|stage-prefetch-profile) ready=6;;
        stage-pair|stage-pair-profile) ready=5;;
        stage-stream|stage-stream-profile) ready=7;;
        plan-compact|plan-compact-profile) ready=8;;
        wqe-prep|wqe-prep-profile) ready=9;;
        wqe-prefetch|wqe-prefetch-profile) ready=11;;
        input6|input6-profile) ready=12;;
        input-ready|input-ready-profile) ready=13;;
        input-pair|input-pair-profile) ready=14;;
        input-pipe|input-pipe-profile) ready=15;;
        input-stream|input-stream-profile) ready=16;;
        input-front|input-front-profile) ready=17;;
        mte-pack|mte-pack-profile) ready=10;;
        ready-r32|ready-r32-profile) ready=4; recv=32;;
        ready-r36|ready-r36-profile) ready=4; recv=36;;
        ready-pipeline|ready-pipeline-profile) ready=3; defer=ON;;
        ready-shared|ready-shared-profile) ready=2;;
        notify-simt|notify-simt-profile) notify=1; remote=ON;;
        notify-core|notify-core-profile) notify=2; remote=ON;;
        card-all|card-all-profile) coalesce=1;;
        card-row|card-row-profile) coalesce=1; coalesce_rows=1;;
        server-all|server-all-profile) coalesce=2;;
        server-row|server-row-profile) coalesce=2; coalesce_rows=1;;
        batch|batch-profile) local_first=ON; batch_final=ON;;
        batch-pipeline|batch-pipeline-profile) local_first=ON; batch_final=ON; prefetch_return=ON;;
        local-first|local-first-profile) local_first=ON;;
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
         --ready "$ready" --notify "$notify" --coalesce "$coalesce" --coalesce-rows "$coalesce_rows"
         --recv-cores "$recv" --early-reduce "$early" --remote-first "$remote" --defer-return "$defer"
         --early-issue "$issue" --progress-probe "$probe" --simt-meta "$simt_meta" --simt-flag "$simt_flag"
         --prefetch-return "$prefetch_return" --local-first "$local_first" --batch-final "$batch_final" --poll-nop "$poll_nop" --ranks-per-host "$ranks_per_host")
    printf '\nBuilding %s\n' "$name"
    if ((dry)); then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
printf '\nBuild selection complete: %s\n' "$variants"
