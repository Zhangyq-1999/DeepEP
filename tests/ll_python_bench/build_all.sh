#!/usr/bin/env bash
# Install only the current clean comparisons into the selected conda environment.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
variants=$ll_bench_clean_variants
args=()
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'build_all.sh [--variants base,input6,input6-l2,input6-r32,input6-remote] [build.sh environment options]'
            echo 'Defaults: five clean Combine variants installed in py311_zyq; Dispatch is shared.'
            exit 0;;
        --variants)
            (($#>=2)) || { echo 'Missing --variants value' >&2; exit 2; }
            variants=$2; shift 2;;
        --conda-env|--ranks-per-host|--jobs|--cann-home|--shmem-root|--mpi-home)
            (($#>=2)) || { echo "Missing value: $1" >&2; exit 2; }
            args+=("$1" "$2"); shift 2;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
[[ $variants =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || { echo 'Provide comma-separated clean variants' >&2; exit 2; }
IFS=, read -r -a names <<< "$variants"
for name in "${names[@]}"; do
    case "$name" in
        base|ready-snapshot|wqe-prep|wqe-prefetch|input6|input-ready|input-pair|input-pipe|input-stream|input-front|input-interleave|input-l2|input6-l2|input6-r32|input6-remote|input6-fused|mte-pack|card-all|card-row|server-all|server-row) ;;
        *) echo "Unknown clean variant: $name" >&2; exit 2;;
    esac
done
for name in "${names[@]}"; do
    bash "$root/tests/ll_python_bench/build.sh" --variant "$name" "${args[@]}"
done
