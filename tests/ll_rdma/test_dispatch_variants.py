"""CPU-only launch selection and ABI checks; never starts an NPU."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dispatch_variant_mapping_and_invalid_inputs(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if compiler is None and not zig:
        pytest.skip("C++17 compiler required for the CPU dispatch selector fixture")
    command = [zig, "c++"] if zig else [compiler]
    source = tmp_path / "dispatch_variants.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "csrc/kernels/dispatch/low_latency/common/ll_dispatch_variant.h"
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_input.h"
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
using namespace DispatchDedup;
// Independent table: dtype, quant mode, output dtype, input mode.
constexpr uint8_t formats[7][4] = {
    {15,4,15,2}, {3,0,3,0}, {3,4,15,1}, {3,4,16,1},
    {11,0,11,0}, {11,4,15,1}, {11,4,16,1}
};
uint32_t expected(const uint8_t* fields, uint8_t group) {
    if (group==2) {
        if(fields[1]!=0 || fields[3]!=0 || (fields[2]!=fields[0] && fields[2]!=255))return 0xffffffffU;
        return fields[0]==3?14:(fields[0]==11?15:0xffffffffU);
    }
    if (group != 4 && group != 16) return 0xffffffffU;
    for (unsigned mode=0; mode<7; ++mode) {
        bool matches = true;
        for (unsigned field=0; field<4; ++field) {
            // Plain16's reserved output dtype means the input dtype.
            if (field == 2 && fields[field] == 255 && (mode == 1 || mode == 4)) continue;
            if (fields[field] != formats[mode][field]) matches = false;
        }
        if (matches) return mode*2 + (group==16);
    }
    return 0xffffffffU;
}
void check(const uint8_t* fields, uint8_t group) {
    assert(LlDispatchVariantForTiling(fields[0],fields[1],fields[2],fields[3],group)
           == expected(fields, group));
}
int main() {
    static_assert(sizeof(deepep::EpServerDedupKernelArgs)==152);
    static_assert(kLlDispatchVariantCount==16);
    static_assert(kLlDispatchFp16==DISPATCH_DEDUP_DATA_TYPE_FP16);
    static_assert(kLlDispatchBf16==DISPATCH_DEDUP_DATA_TYPE_BFP16);
    static_assert(kLlDispatchE4M3==DISPATCH_DEDUP_DATA_TYPE_FP8E4M3);
    static_assert(kLlDispatchE5M2==DISPATCH_DEDUP_DATA_TYPE_FP8E5M2);
    static_assert(kLlDispatchPlain16==uint8_t(EpServerDedupInputMode::Plain16));
    static_assert(kLlDispatchQuantize16==uint8_t(EpServerDedupInputMode::Quantize16));
    static_assert(kLlDispatchPrequantized==uint8_t(EpServerDedupInputMode::PrequantizedFp8Packs));
    static_assert(LlDispatchVariantForTiling(15,4,15,2,4)==0);
    static_assert(LlDispatchVariantForTiling(11,4,16,1,16)==13);
    for (unsigned mode=0; mode<7; ++mode) {
        for (unsigned group=0; group<256; ++group) check(formats[mode],uint8_t(group));
        for (uint8_t group : {uint8_t(4), uint8_t(16)}) {
            assert(LlDispatchVariantForTiling(formats[mode][0],formats[mode][1],
                   formats[mode][2],formats[mode][3],group)==mode*2+(group==16));
            for (unsigned field=0; field<4; ++field) for (unsigned value=0; value<256; ++value) {
                uint8_t modified[4];
                for (unsigned i=0;i<4;++i) modified[i]=formats[mode][i];
                modified[field]=uint8_t(value);
                check(modified,group);
            }
        }
    }
    const uint8_t dtypes[]={0,3,11,15,16,255};
    const uint8_t quants[]={0,1,4,255};
    const uint8_t modes[]={0,1,2,3,255};
    for (uint8_t dtype:dtypes) for (uint8_t output:dtypes)
        for (uint8_t quant:quants) for (uint8_t input:modes) {
            uint8_t fields[]={dtype,quant,output,input};
            check(fields,2); check(fields,4); check(fields,16); check(fields,8);
        }
}
''', encoding="utf-8")
    executable = tmp_path / ("dispatch_variants.exe" if os.name == "nt" else "dispatch_variants")
    subprocess.run([*command, "-std=c++17", "-UNDEBUG", "-iquote", str(ROOT), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_host_launch_uses_same_format_builder_as_tiling():
    host = (ROOT / "csrc/ops/low_latency.cpp").read_text(encoding="utf-8")
    assert "auto data = dispatch_format_tiling(p, dtype);" in host
    assert "dispatch_format_tiling(session.world(), dtype)" in host
    assert "dispatch_server_dedup_kernel_do_v2(reinterpret_cast<void*>(stream_address), &args, variant)" in host
    launch = (ROOT / "csrc/kernels/dispatch/low_latency/launch.hpp").read_text(encoding="utf-8")
    assert "dispatch_server_dedup_kernel_do_v2(" in launch
    assert "dispatch_server_dedup_kernel_do(" not in launch


@pytest.mark.parametrize("reference_enabled", [False, True])
def test_real_dispatch_launch_routes_exactly_one_specialization(tmp_path, reference_enabled):
    compiler = shutil.which("g++") or shutil.which("clang++")
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if compiler is None and not zig:
        pytest.skip("C++17 compiler required for the real Host launch fixture")
    command = [zig, "c++"] if zig else [compiler]
    source = tmp_path / "dispatch_launch.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include "csrc/kernels/dispatch/low_latency/launch.hpp"
#include "csrc/kernels/dispatch/low_latency/common/ll_group_policy.h"
static unsigned calls = 0, calledMode = 99, calledGroup = 0;
static void* expectedStream = nullptr;
static const deepep::EpServerDedupKernelArgs* expectedArgs = nullptr;
void record(unsigned mode, unsigned group, void* stream, const deepep::EpServerDedupKernelArgs* args) {
    ++calls;
    calledMode = mode;
    calledGroup = group;
    assert(stream == expectedStream);
    assert(args == expectedArgs);
}
#define STUB(name, mode, group) \
extern "C" void dispatch_server_dedup_launch_##name(void* stream, const deepep::EpServerDedupKernelArgs* args) \
{ record(mode, group, stream, args); }
STUB(prequant_e4m3_g4, 0, 4)
STUB(prequant_e4m3_g16, 0, 16)
STUB(fp16_g4, 1, 4)
STUB(fp16_g16, 1, 16)
STUB(fp16_e4m3_g4, 2, 4)
STUB(fp16_e4m3_g16, 2, 16)
STUB(fp16_e5m2_g4, 3, 4)
STUB(fp16_e5m2_g16, 3, 16)
STUB(bf16_g4, 4, 4)
STUB(bf16_g16, 4, 16)
STUB(bf16_e4m3_g4, 5, 4)
STUB(bf16_e4m3_g16, 5, 16)
STUB(bf16_e5m2_g4, 6, 4)
STUB(bf16_e5m2_g16, 6, 16)
STUB(fp16_g2, 1, 2)
STUB(bf16_g2, 4, 2)
#undef STUB
void rejects(const deepep::EpServerDedupKernelArgs* args, uint32_t variant) {
    calls = 0;
    bool rejected = false;
    try { dispatch_server_dedup_kernel_do_v2(expectedStream, args, variant); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && calls == 0);
}
int main() {
    static_assert(sizeof(deepep::EpServerDedupKernelArgs) == 152);
    assert(deepep_ll_core_protocol_version() == 41008);
    assert(deepep_ll_rdma_enabled() == 1);
    assert(deepep_ll_stage_timer_enabled() == 0);
#ifdef DEEPEP_LL_REFERENCE_PROFILE
    assert(deepep_ll_reference_profile_enabled() == 1);
#else
    assert(deepep_ll_reference_profile_enabled() == 0);
#endif
    // The Host query and the device policy must select the same specialization,
    // including the EP2/4/8 standalone benchmark and the EP16/32 boundary.
    for (uint32_t world = 1U; world <= 256U; ++world)
        assert(deepep_ll_ranks_per_group(world) == DispatchDedup::LlGroupForWorld(world));
    deepep::EpServerDedupKernelArgs args{};
    int streamHandle = 1;
    expectedStream = &streamHandle;
    expectedArgs = &args;
    for (uint32_t variant = 0; variant < 14; ++variant) {
        calls = 0;
        calledMode = 99;
        calledGroup = 0;
        dispatch_server_dedup_kernel_do_v2(expectedStream, &args, variant);
        assert(calls == 1 && calledMode == variant / 2 && calledGroup == (variant % 2 ? 16U : 4U));
        rejects(nullptr, variant);
    }
    for (uint32_t variant : {14U, 15U}) {
        calls=0;
        dispatch_server_dedup_kernel_do_v2(expectedStream,&args,variant);
        assert(calls==1 && calledGroup==2 && calledMode==(variant==14?1:4));
    }
    assert(deepep_ll_ranks_per_group(2)==2);
    for (uint32_t variant : {16U, 17U, 255U, 0xffffffffU}) {
        rejects(&args, variant);
        rejects(nullptr, variant);
    }
}
''', encoding="utf-8")
    executable = tmp_path / ("dispatch_launch.exe" if os.name == "nt" else "dispatch_launch")
    flags = ["-DCATLASS_ARCH=3510", "-DDEEPEP_LL_RDMA=1"]
    if reference_enabled:
        flags.append("-DDEEPEP_LL_REFERENCE_PROFILE=1")
    subprocess.run([
        *command, "-std=c++17", "-UNDEBUG", *flags, "-iquote", str(ROOT), str(source),
        str(ROOT / "csrc/kernels/dispatch/low_latency.cpp"), "-o", str(executable),
    ], check=True)
    subprocess.run([str(executable)], check=True)
