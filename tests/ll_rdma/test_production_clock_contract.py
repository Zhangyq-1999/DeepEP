"""Keep production LL polling independent of diagnostic cycle counters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cycle_clock_is_confined_to_explicit_timer_headers():
    diagnostic = {"ll_reference_profile_device.h", "combine_profile_device.h"}
    for directory in (ROOT / "csrc/kernels/dispatch/low_latency",
                      ROOT / "csrc/kernels/combine/low_latency"):
        for source in directory.rglob("*"):
            if source.suffix not in {".h", ".hpp", ".cpp"}:
                continue
            text = source.read_text(encoding="utf-8")
            assert "PollDeadline" not in text, source
            if source.name not in diagnostic:
                assert "GetSystemCycle" not in text, source
            else:
                guard = ("DEEPEP_LL_REFERENCE_PROFILE" if source.name == "ll_reference_profile_device.h"
                         else "DEEPEP_LL_COMBINE_PROFILE")
                assert f"#ifdef {guard}" in text, source


def test_ll_build_keeps_legacy_ht_and_barrier_clock_paths_out():
    cmake = (ROOT / "cmake/device/CMakeLists.txt").read_text(encoding="utf-8")
    sources = cmake.split("set(_ll_sources", 1)[1].split("add_library(deepep_kernel_ll_plain", 1)[0]
    assert 'csrc/kernels/dispatch/low_latency.cpp' in sources
    assert '${DEEPEP_LL_COMBINE_SOURCE}' in sources
    assert 'csrc/kernels/runtime/host_barrier.cpp' not in sources
    assert 'csrc/kernels/dispatch/dispatch.cpp' not in sources
    assert 'csrc/kernels/combine/combine.cpp' not in sources
