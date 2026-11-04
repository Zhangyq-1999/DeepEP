"""Execute the benchmark's actual pre-kernel checks with its exported queries."""
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["plain", "early", "local", "batch", "card-all", "card-row", "server-all", "server-row", "notify-simt", "notify-core", "ready-task", "ready-shared"])
@pytest.mark.parametrize("profile", [False, True])
def test_benchmark_build_handshake(tmp_path, mode, profile):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU checks")
    device = (ROOT / "csrc/kernels/combine/low_latency.cpp").read_text(encoding="utf-8")
    queries = "\n".join(re.findall(
        r'extern "C" DEEPEP_EXPORT (?:uint32_t|int32_t) deepep_ll_combine_\w+\(\)\n\{.*?\n\}',
        device, re.S))
    assert "deepep_ll_combine_early_reduce_enabled" in queries
    main = (ROOT / "tests/ll_cpp_bench/main.cpp").read_text(encoding="utf-8")
    start = main.index("    const auto recv = CombineDedup::EpCombineRecvCoreCount(cores);")
    guard = main[start:main.index("    if (!profileBuild)", start)]
    # This is the production guard, not a reimplementation of its logic. The
    # local-first case reproduced the false mismatch before any kernel launch.
    source = tmp_path / "handshake.cpp"
    source.write_text('''#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
#define DEEPEP_EXPORT
''' + queries + '''
void Check(uint32_t cores) {
''' + guard + '''
}
int main() {
    Check(56);
    bool rejected=false;
    try { Check(0); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
''', encoding="utf-8")
    flags = []
    if mode in ("early", "local", "batch"):
        flags.append("-DDEEPEP_LL_COMBINE_" + ("EARLY_REDUCE" if mode == "early" else "LOCAL_FIRST") + "=1")
    if mode == "batch":
        flags.append("-DDEEPEP_LL_COMBINE_BATCH_FINAL=1")
    if profile:
        flags.append("-DDEEPEP_LL_COMBINE_PROFILE=1")
    coalesce = 1 if mode.startswith("card-") else 2 if mode.startswith("server-") else 0
    rows = int(mode.endswith("-row"))
    notify = 1 if mode == "notify-simt" else 2 if mode == "notify-core" else 0
    ready = 1 if mode == "ready-task" else 2 if mode == "ready-shared" else 0
    flags.extend([f"-DDEEPEP_LL_COMBINE_READY={ready}", f"-DDEEPEP_BENCH_READY={ready}"])
    flags.extend([f"-DDEEPEP_LL_COMBINE_NOTIFY={notify}", f"-DDEEPEP_BENCH_NOTIFY={notify}"])
    flags.extend([f"-DDEEPEP_LL_COMBINE_COALESCE={coalesce}", f"-DDEEPEP_BENCH_COALESCE={coalesce}",
                  f"-DDEEPEP_LL_COMBINE_COALESCE_ROWS={rows}", f"-DDEEPEP_BENCH_COALESCE_ROWS={rows}"])
    for name in ("REMOTE_FIRST", "DEFER_RETURN", "EARLY_ISSUE", "PROGRESS_PROBE",
                 "SIMT_META", "SIMT_FLAG", "PREFETCH_RETURN", "POLL_NOP", "LOCAL_FIRST", "BATCH_FINAL"):
        enabled = (name == "LOCAL_FIRST" and mode in ("local", "batch")) or (name == "BATCH_FINAL" and mode == "batch")
        flags.append(f"-DDEEPEP_BENCH_{name}={int(enabled)}")
    exe = tmp_path / "handshake.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", *flags,
                    "-I" + str(ROOT / "csrc"), str(source), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=30)
