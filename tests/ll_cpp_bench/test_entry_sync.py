"""CPU protocol/queue tests. These do not validate CANN codegen or NPU transport."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def compile_run(tmp_path, source, *flags):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    executable = tmp_path / "entry.exe"
    subprocess.run([zig, "c++", "-std=c++20", "-O1", "-UNDEBUG", *flags,
                    str(source), "-o", str(executable)], check=True, timeout=180)
    subprocess.run([str(executable)], check=True, timeout=120)


def test_mixed_transport_repeated_barriers(tmp_path):
    compile_run(tmp_path, HERE / "entry_sync_device_test.cpp", "-pthread")


def test_notify_submission_order_and_cleanup(tmp_path):
    # Compile the production RAII helper with a queue-aware Runtime substitute.
    shutil.copyfile(HERE / "entry_sync.hpp", tmp_path / "entry_sync.hpp")
    (tmp_path / "variant_api.hpp").write_text(r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
using aclrtNotify=void*;
using aclrtStream=void*;
inline int storage[3], notifyCount=0, streamCount=0, hostBarriers=0;
inline std::vector<std::string> queue;
inline bool armed=false, released=false, failRelease=false;
constexpr int MPI_COMM_WORLD=1;
inline void MPI_Abort(int,int) { assert(false); }
inline int aclrtCreateNotify(void** p,int) { *p=&storage[0]; ++notifyCount; return 0; }
inline int aclrtCreateStream(void** p) { *p=&storage[1]; ++streamCount; return 0; }
inline int aclrtWaitAndResetNotify(void*,void*,unsigned timeout) {
    assert(!armed && timeout==10000); armed=true; released=false; queue.push_back("wait"); return 0;
}
inline int aclrtRecordNotify(void*,void* stream) {
    assert(stream==&storage[1] && armed);
    if(failRelease) { failRelease=false; return 7; }
    released=true; queue.push_back("release"); return 0;
}
inline int aclrtSynchronizeStreamWithTimeout(void*,int) { assert(!armed); return 0; }
inline int aclrtDestroyNotify(void*) { assert(!armed); --notifyCount; return 0; }
inline int aclrtDestroyStream(void*) { assert(!armed); --streamCount; return 0; }
namespace llbench {
struct Options { std::string entrySync="device"; unsigned kernelTimeout=10; };
struct Coordinator { void barrier() const { ++hostBarriers; } };
struct Runtime {
    Options options;
    const Coordinator* peers;
    aclrtStream stream=&storage[2];
    uint64_t ffts=42;
    void synchronize(const char*) const {
        assert(!armed || released); armed=false; queue.push_back("complete");
    }
};
}
''')
    source = tmp_path / "test.cpp"
    source.write_text(r'''
#include "entry_sync.hpp"
extern "C" void ll_bench_entry_sync_do(void* s,uint64_t ffts,uint64_t ticks) {
    assert(s==&storage[2] && ffts==42 && ticks==10000000000ULL);
    assert(armed && !released); queue.push_back("device-sync");
}
int main() {
    llbench::Coordinator peers;
    llbench::Runtime r{{},&peers};
    {
        llbench::EntrySync sync(r);
        for(int i=0;i<3;++i) {
            queue.clear(); sync.before();
            assert(!released);
            queue.insert(queue.end(),{"start","combine","end"});
            sync.submitted(); r.synchronize("sample");
            assert(queue==std::vector<std::string>({"wait","device-sync","start","combine","end","release","complete"}));
        }
    }
    assert(!hostBarriers && !notifyCount && !streamCount);
    r.options.entrySync="host";
    { llbench::EntrySync sync(r); sync.before(); sync.submitted(); }
    assert(hostBarriers==1 && !notifyCount && !streamCount);
    r.options.entrySync="device";
    // An exception between arming and release must not destroy a live Notify.
    { llbench::EntrySync sync(r); sync.before(); }
    { llbench::EntrySync sync(r); sync.before(); failRelease=true;
      try { sync.submitted(); assert(false); } catch(const std::runtime_error&) {} }
    // Release succeeded but the caller unwinds before its normal stream wait.
    { llbench::EntrySync sync(r); sync.before(); sync.submitted(); }
    assert(!armed && !notifyCount && !streamCount && hostBarriers==1);
}
''')
    compile_run(tmp_path, source)
