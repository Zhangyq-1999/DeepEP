#!/usr/bin/env bash
# Install clean comparisons and independent diagnostic builds in one environment.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
variants=$ll_bench_clean_variants
profile=2
args=()
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'build_all.sh [--variants base,input6-l2-route-only,ops-shmem-overlap,ops-shmem-cq,ops-shmem-batch] [--profile 0|1|2] [build.sh environment options]'
            echo 'Defaults: five clean + two diagnostic variants in py311_zyq; Dispatch is shared.'
            exit 0;;
        --variants)
            (($#>=2)) || { echo 'Missing --variants value' >&2; exit 2; }
            variants=$2; shift 2;;
        --profile)
            (($#>=2)) || { echo 'Missing --profile value' >&2; exit 2; }
            profile=$2; shift 2;;
        --conda-env|--ranks-per-host|--jobs|--cann-home|--shmem-root|--mpi-home)
            (($#>=2)) || { echo "Missing value: $1" >&2; exit 2; }
            args+=("$1" "$2"); shift 2;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
[[ $variants =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || { echo 'Provide comma-separated clean variants' >&2; exit 2; }
[[ $profile =~ ^[012]$ ]] || { echo 'Use --profile 0, 1 or 2' >&2; exit 2; }
IFS=, read -r -a names <<< "$variants"
for name in "${names[@]}"; do
    case "$name" in
        base|batch|batch-i64|input6-l2-route-i64|ops-shmem-base|ops-shmem-retire|ops-shmem-overlap|ops-shmem-cq|ops-shmem-batch|ready-snapshot|wqe-prep|wqe-prefetch|input6|input-ready|input-pair|input-pipe|input-stream|input-front|input-interleave|input-l2|input6-l2|input6-l2-stage|input6-l2-tx|input6-l2-rx|input6-l2-all|input6-l2-pipe|input6-l2-setup|input6-l2-single|input6-l2-lean|input6-l2-route|input6-l2-route-only|input6-l2-tiny-setup|input6-l2-tiny-probe|input6-l2-tiny|input6-l2-route-priority|input6-l2-route-tx-bypass|input6-l2-route-plan|input6-l2-remote|input6-r32|input6-remote|input6-fused|input6-rpipe|input6-rfull|mte-pack|card-all|card-row|server-all|server-row) ;;
        *) echo "Unknown clean variant: $name" >&2; exit 2;;
    esac
done
if ((profile)); then
    profiles=()
    for name in "${names[@]}"; do
        [[ $variants != "$ll_bench_clean_variants" || $name == ops-shmem-cq || $name == ops-shmem-batch ]] || continue
        if [[ $name == base ]]; then profiles+=(ref-profile); else profiles+=("$name-profile"); fi
    done
    names+=("${profiles[@]}")
fi
for name in "${names[@]}"; do
    bash "$root/tests/ll_python_bench/build.sh" --variant "$name" "${args[@]}"
done
