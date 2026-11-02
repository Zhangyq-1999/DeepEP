#!/usr/bin/env bash
# Host-only CSV analysis, also works with results produced before this script.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if (($# != 1 && $# != 2)); then
    echo 'Usage: summarize.sh PHASE_DIRECTORY | --suite RUN_DIRECTORY' >&2
    exit 2
fi
source_file="$root/tests/ll_cpp_bench/profile_report.cpp"
build="$root/build/ll-cpp-report"
binary="$build/ll_combine_profile_report"
mkdir -p "$build"
if [[ ! -x "$binary" || "$source_file" -nt "$binary" ||
      "$root/tests/ll_cpp_bench/profile.hpp" -nt "$binary" ||
      "$root/tests/ll_cpp_bench/compact_report.hpp" -nt "$binary" ||
      "$root/csrc/kernels/combine/low_latency/combine_profile_layout.h" -nt "$binary" ]]; then
    # Use a Host compiler, independent of the CANN/Conda/MPI environment.
    compiler=${HOST_CXX:-g++}
    temporary=$(mktemp "$build/.report-XXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    "$compiler" -std=c++17 -O2 "$source_file" -o "$temporary"
    chmod +x "$temporary"
    mv -f -- "$temporary" "$binary"
fi
"$binary" "$@"
