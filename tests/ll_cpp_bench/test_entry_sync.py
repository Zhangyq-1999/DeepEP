"""Protocol and enqueue tests; SDK primitives are mocked, not device validated."""
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def compile_test(tmp_path, source):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    exe = tmp_path / "entry-test.exe"
    subprocess.run([zig, "c++", "-std=c++20", "-O1", "-UNDEBUG", "-I", str(tmp_path), "-I", str(HERE),
                    str(source), "-o", str(exe)], check=True, timeout=180)
    return exe


def test_repeated_mixed_transport_barrier(tmp_path):
    exe = compile_test(tmp_path, HERE / "entry_sync_device_test.cpp")
    subprocess.run([str(exe)], check=True, timeout=60)


def test_entry_gate_orders_barrier_events_and_launch(tmp_path):
    # Copy only the real wrapper so its dependency resolves to the Runtime mock.
    (tmp_path / "entry_sync.hpp").write_text((HERE / "entry_sync.hpp").read_text())
    (tmp_path / "variant_api.hpp").write_text(r'''
#pragma once
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
using aclrtNotify = void*; using aclrtStream = void*;
inline std::vector<std::string> calls;
inline int hostBarriers=0, handles=0;
inline bool armed=false, released=false, failRelease=false;
constexpr int MPI_COMM_WORLD=1;
inline int MPI_Abort(int,int) { std::_Exit(19); }
inline int aclrtCreateNotify(void** p,int) { *p=(void*)1; ++handles; return 0; }
inline int aclrtCreateStream(void** p) { *p=(void*)2; ++handles; return 0; }
inline int aclrtDestroyNotify(void*) { assert(!armed); --handles; return 0; }
inline int aclrtDestroyStream(void*) { assert(!armed); --handles; return 0; }
inline int aclrtWaitAndResetNotify(void*,void* stream,unsigned timeout) {
    assert(stream==(void*)3 && timeout==10000 && !armed);
    armed=true; released=false; calls.push_back("wait"); return 0;
}
inline int aclrtRecordNotify(void*,void* stream) {
    assert(stream==(void*)2 && armed && !released);
    if(failRelease) return 7;
    released=true; calls.push_back("release"); return 0;
}
inline int aclrtSynchronizeStreamWithTimeout(void*,int) { assert(!armed); return 0; }
namespace llbench {
struct Options { std::string entrySync="device"; unsigned kernelTimeout=10, rank=0, device=12; };
struct Coordinator { void barrier() const { ++hostBarriers; calls.push_back("mpi"); } };
struct Runtime {
    Options options; const Coordinator* peers; aclrtStream stream=(void*)3; uint64_t ffts=42;
    void synchronize(const char*,unsigned=0,unsigned=0) const {
        assert(!armed || released); armed=false; calls.push_back("complete");
    }
};
}
''')
    source = tmp_path / "test.cpp"
    source.write_text(r'''
#include "entry_sync.hpp"
extern "C" void ll_bench_entry_sync_do(void* stream,uint64_t ffts) {
    assert(stream==(void*)3 && ffts==42 && armed && !released);
    calls.push_back("device-sync");
}
int main(int argc,char**) {
    llbench::Coordinator peers;
    llbench::Runtime runtime{{},&peers};
    auto work=[] { calls.insert(calls.end(),{"start","combine","end"}); };
    if(argc>1) {
        llbench::EntrySync sync(runtime);
        if(argc==2) sync.enqueue([] { throw std::runtime_error("submit failed"); });
        else { failRelease=true; sync.enqueue(work); }
        assert(false);
    }
    {
        llbench::EntrySync sync(runtime);
        for(int i=0;i<3;++i) {
            calls.clear(); sync.enqueue(work); sync.complete("sample");
            assert(calls==std::vector<std::string>({"wait","device-sync","start","combine","end","release","complete"}));
        }
        assert(hostBarriers==0 && handles==2);
    }
    assert(handles==0);
    runtime.options.entrySync="host";
    { llbench::EntrySync sync(runtime); calls.clear(); sync.enqueue(work); sync.complete("sample"); }
    assert(calls==std::vector<std::string>({"mpi","start","combine","end","complete"}));
    assert(hostBarriers==1 && handles==0);
    runtime.options.entrySync="device";
    { llbench::EntrySync sync(runtime); sync.enqueue(work); } // destructor retires released work
    assert(!armed && handles==0);
}
''')
    exe = compile_test(tmp_path, source)
    subprocess.run([str(exe)], check=True, timeout=20)
    for args in (["submit"], ["release", "fail"]):
        result = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=20)
        assert result.returncode == 19 and "FAIL rank=0 device=12 entry_sync=device" in result.stderr
