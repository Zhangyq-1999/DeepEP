"""Check diagnostic record isolation, OFF erasure and Host CSV decoding on CPU."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("enabled", [False, True])
def test_profile_record_and_csv(tmp_path, enabled):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU C++ checks")
    (tmp_path / "kernel_operator.h").write_text(r'''
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
inline uint32_t core = 0;
#ifdef DEEPEP_LL_COMBINE_PROFILE
inline uint32_t GetBlockIdx() { return core; }
inline uint64_t tick = uint64_t(1) << 60, clocks = 0, fences = 0;
inline uint64_t GetSystemCycle() { ++clocks; return tick += 17; }
enum class MemDsbT { DDR };
template<MemDsbT> void DataSyncBarrier() { ++fences; }
#endif
}
''')
    (tmp_path / "shmem.h").write_text(r'''
#pragma once
#include <cassert>
inline uint8_t* flushed = nullptr;
inline void dcci_cachelines(uint8_t* p, uint64_t n) { assert(n == CombineProfile::kWords * 8 - 1); flushed = p; }
''')
    source = tmp_path / "profile.cpp"
    source.write_text(r'''
#include "kernels/combine/low_latency/combine_profile_device.h"
#include "profile.hpp"
#include <algorithm>
#include <cassert>
#include <sstream>
using namespace CombineProfile;
int main() {
    constexpr uint64_t guard = 0x5a5a5a5a5a5a5a5aULL;
    for (uint32_t level : {0U, 1U, 2U}) {
        alignas(512) uint64_t raw[58 * kWords];
        std::fill_n(raw, 58 * kWords, guard);
        auto base = raw + kWords;
        std::fill_n(base, 56 * kWords, 0);
        for (uint32_t core = 0; core < 56; ++core) {
            AscendC::core = core;
            DeviceProfile profile;
            View view;
            view.Bind(&profile);
            profile.Init(reinterpret_cast<uint8_t*>(base), level);
            view.Mark(StageBegin);
            view.Set(StageRows, 7);
            const auto start = view.DetailClock();
            view.Elapsed(ProbeTicks, start);
            view.Add(ProbeCalls, 3);
            view.Mark(StageEnd);
            view.MarkOnce(DirectFirstSubmit);
            view.MarkOnce(DirectFirstSubmit);
            const auto submit = view.DetailClock();
            view.SampleMax(DirectSubmitTicks, DirectSubmitMaxTicks, DirectSubmitMaxPeer, DirectSubmitMaxToken,
                           submit, 3, 48);
            view.Mark(FlushDone);
            auto row = base + core * kWords;
            for (uint32_t i = 0; i < kWords; ++i) assert(row[i] == 0); // one tail export
            profile.Finish(3, 4, 16);
#ifdef DEEPEP_LL_COMBINE_PROFILE
            if (level) {
                assert(row[Schema] == kSchema && row[Core] == core && row[StageRows] == 7);
                assert(row[ProbeTicks] == (level == 2 ? 17U : 0U));
                assert(row[ProbeCalls] == (level == 2 ? 3U : 0U));
                assert(row[DirectSubmitTicks] == (level == 2 ? 17U : 0U));
                assert(row[DirectSubmitMaxPeer] == (level == 2 ? 3U : 0U));
                assert(row[DirectSubmitMaxToken] == (level == 2 ? 48U : 0U));
                assert(flushed == reinterpret_cast<uint8_t*>(row));
                assert(row[Entry] > (uint64_t(1) << 60));
            } else {
                for (uint32_t i = 0; i < kWords; ++i) assert(row[i] == 0);
            }
#else
            static_assert(sizeof(DeviceProfile) == 1 && sizeof(View) == 1);
            assert(start == 0);
            for (uint32_t i = 0; i < kWords; ++i) assert(row[i] == 0);
#endif
        }
        for (uint32_t i = 0; i < kWords; ++i) assert(raw[i] == guard && raw[57*kWords+i] == guard);
#ifdef DEEPEP_LL_COMBINE_PROFILE
        if (level) {
            std::vector<uint64_t> data(base, base + 56*kWords);
            std::ostringstream header, rows;
            llbench::profile_header(header);
            llbench::profile_rows(rows, data, 56, 28, 3, 4, 16, 15, level, 0);
            std::istringstream lines(rows.str());
            std::string line;
                const auto h = header.str();
                const auto columns = std::count(h.begin(), h.end(), ',');
            uint32_t n = 0;
            while (std::getline(lines, line)) {
                assert(std::count(line.begin(), line.end(), ',') == columns);
                assert(line.find(n < 28 ? "send" : "receive") != std::string::npos);
                ++n;
            }
            assert(n == 56);
            assert(llbench::profile_interval(data.data(), Entry, FlushDone) > 0);
            assert(llbench::profile_interval(data.data(), PlanBegin, PlanEnd) == 0);
            data[Complete] = 0;
            bool rejected = false;
            try { llbench::profile_rows(rows, data, 56, 28, 3, 4, 16, 15, level, 0); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected);
        }
#endif
    }
    DeviceProfile disabled;
    disabled.Init(nullptr, 2);
    disabled.Mark(FlushDone);
    assert(disabled.DetailClock() == 0);
    disabled.Finish(0, 4, 16);
#ifdef DEEPEP_LL_COMBINE_PROFILE
    AscendC::core = 0;
    uint64_t sample[kWords]{};
    DeviceProfile probes;
    probes.Init(reinterpret_cast<uint8_t*>(sample), 2);
    probes.Probe(false, 32, 0xffffffffU, 120, probes.DetailClock());
    probes.Probe(true, 48, 2, 15, probes.DetailClock());
    probes.Mark(FlushDone);
    probes.Finish(0, 4, 72);
    assert(sample[ProbeCalls] == 2 && sample[ProbeMisses] == 1);
    assert(sample[ProbeTicks] == 34 && sample[ProbeMissTicks] == 17);
    assert(sample[ProbeFlagBlocks] == 135);
    assert(sample[ProbeMaxToken] == 32 && sample[ProbeMaxSlot] == 0xffffffffU);
    assert(sample[FinalFirstReady] && sample[FinalLastReady] >= sample[FinalFirstReady]);
#endif
}
''')
    exe = tmp_path / "profile.exe"
    flags = ["-DDEEPEP_LL_COMBINE_PROFILE=1"] if enabled else []
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG", *flags,
                    "-I" + str(tmp_path), "-I" + str(ROOT / "csrc"),
                    "-I" + str(Path(__file__).parent), str(source), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)
