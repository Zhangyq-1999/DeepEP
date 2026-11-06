"""Load two real pybind modules with CPU ACL/SHMEM substitutes; share one session."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest

HERE = Path(__file__).resolve().parent


def test_native_modules_share_session_and_keep_implementations_separate(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig or os.name != "nt":
        pytest.skip("Windows host test requires DEEPEP_TEST_ZIG and Python import library")
    pybind11 = pytest.importorskip("pybind11")
    (tmp_path/"acl").mkdir()
    (tmp_path/"runtime").mkdir()
    (tmp_path/"acl/acl.h").write_text(r'''
#pragma once
#include <cstdint>
#include <cstring>
inline unsigned resets=0;
using aclrtStream=void*; using aclrtNotify=void*;
inline int aclrtCreateNotify(void** p,int) {*p=(void*)1;return 0;}
inline int aclrtCreateStream(void** p) {*p=(void*)2;return 0;}
inline int aclrtDestroyNotify(void*) {return 0;}
inline int aclrtDestroyStream(void*) {return 0;}
inline int aclrtWaitAndResetNotify(void*,void*,unsigned) {return 0;}
inline int aclrtRecordNotify(void*,void*) {return 0;}
inline int aclrtSynchronizeStreamWithTimeout(void*,int) {return 0;}
inline int aclrtMemset(void* p,size_t capacity,int value,size_t n) {
    if(n!=capacity) throw std::runtime_error("bad reset");
    ++resets; std::memset(p,value,n); return 0;
}
inline int aclrtEventElapsedTime(float* p,int,int) {*p=.01F;return 0;}
''')
    (tmp_path/"runtime/session.hpp").write_text(r'''
#pragma once
namespace deepep {
inline unsigned sessionInits=0, sessionCloses=0;
struct Session {
    unsigned char memory[128]{}; bool closed_=false;
    Session(int,int,uint64_t,const std::string&,unsigned,bool,bool,bool) {++sessionInits;}
    bool closed() const {return closed_;}
    void* workspace() {return memory;}
    void* ll_rdma_stage() {return memory+64;}
    uint64_t capacity() const {return 8ULL<<30;}
    uint32_t ll_rdma_qps() const {return 56;}
    uint64_t workspace_allocation_bytes() const {return sizeof(memory);}
    void close(bool) {++sessionCloses;closed_=true;}
};
struct CoreConfig {unsigned active;};
inline CoreConfig query_ll_core_config() {return {56};}
}
''')
    (tmp_path/"module.cpp").write_text(r'''
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "kernels/combine/low_latency/combine_core_layout.h"
namespace llbench {
constexpr uint64_t workspaceBytes=8ULL<<30;
struct Options {
    unsigned ranksPerHost=4,world=8,hidden=3584,topk=8,experts=384,kernelTimeout=10,timeout=180;
    std::vector<unsigned> tokens{32,48}; std::string dtype="bf16";
    static Options parse(int,char**,int,int,int) {return {};}
    std::string signature() const {return "same";}
};
struct Runtime {
    Options options; void* peers; aclrtStream stream; uint8_t* workspace; uint8_t* rdmaStage;
    uint64_t capacity,ffts; uint32_t cores,rdmaQps; void* progress;
};
}
namespace DispatchDedup {constexpr unsigned kLlPhysicalRanksPerHost=4; struct CommArgs {};}
namespace CombineDedup {inline unsigned EpCombineRecvCoreCount(unsigned n) {return n/2;}}
inline unsigned deepep_ll_combine_profile_enabled() {return 0;}
inline unsigned deepep_ll_combine_return_options() {return TEST_VARIANT;}
inline uint64_t shmemx_get_ffts_config() {return 42;}
struct Event {int value=0; void record(void*) {}}
;
inline unsigned buffers=0,cases=0,launches=0;
struct Buffer {
    uint8_t* p=nullptr;
    explicit Buffer(size_t) {++buffers;}
    ~Buffer() {--buffers;}
    void upload(const void*,size_t) {}
};
struct Case {
    unsigned bs,variant=0; int dispatch=0,combine=0;
    Case(const llbench::Options&,unsigned n,unsigned,void*):bs(n) {++cases;}
    ~Case() {--cases;}
    double precision(bool) {return 0;}
};
inline uint8_t* local_base(const llbench::Runtime& r) {return r.workspace;}
inline DispatchDedup::CommArgs comm_args(const llbench::Runtime& r) {
    if(!r.workspace || !r.rdmaStage || r.ffts!=42 || r.rdmaQps!=56)
        throw std::runtime_error("invalid shared runtime");
    return {};
}
inline void dispatch_server_dedup_kernel_do_v2(void*,void*,unsigned) {}
inline void combine_clear_result_flags_kernel_do(void*,void*,uint64_t,unsigned) {}
inline void combine_server_dedup_kernel_do(void*,void*) {++launches;}
extern "C" void ll_bench_entry_sync_do(void*,uint64_t) {}
#define DEEPEP_BENCH_SOURCE_ID "host-model"
#define DEEPEP_LL_COMBINE_ORIGINAL 0
#include "bindings.hpp"
PYBIND11_MODULE(_ll_py_debug,m) {
    m.def("counts", [] { return py::make_tuple(deepep::sessionInits,deepep::sessionCloses,
                                             resets,buffers,cases,launches); });
}
''')
    def load(path, name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    modules=[]
    for index in range(2):
        path=tmp_path/f"variant{index}.pyd"
        subprocess.run([zig,"c++","-std=c++17","-shared","-O0","-DMS_WIN64",
                        f"-DTEST_VARIANT={index}", f"-DDEEPEP_LL_COMBINE_READY={20 if index else 12}",
                        "-I",str(tmp_path),"-I",str(HERE),"-I",str(HERE.parents[1]/"csrc"),
                        "-I",pybind11.get_include(),"-I",sysconfig.get_path("include"),
                        str(tmp_path/"module.cpp"),
                        str(Path(sys.base_prefix)/"libs"/f"python{sys.version_info.major}{sys.version_info.minor}.lib"),
                        "-o",str(path)],check=True,timeout=180,capture_output=True,text=True)
        modules.append((load(path,f"variant{index}._ll_py_combine"),load(path,f"variant{index}._ll_py_debug")))
    (base,bdebug),(opt,odebug)=modules
    owner=base.Bench(0,8,0,[],123,"tcp://a:1")
    assert bdebug.counts()[:2]==(1,0)
    owner.prepare(32); owner.sample(lambda:None,False); owner.release_case()
    owner.reset_workspace()
    borrowed=opt.Bench(owner.share())
    assert odebug.counts()[:2]==(0,0)
    assert owner.identity()["return_options"]==0 and borrowed.identity()["return_options"]==1
    assert owner.identity()["ready_mode"]==12 and borrowed.identity()["ready_mode"]==20
    assert owner.identity()["core_layout"]==borrowed.identity()["core_layout"]=="halves"
    assert not owner.identity()["control_l2"] and borrowed.identity()["control_l2"]
    assert not owner.identity()["remote_first"] and borrowed.identity()["remote_first"]
    with pytest.raises(RuntimeError,match="Close borrowed"): owner.close()
    with pytest.raises(RuntimeError,match="Release the previous"): owner.reset_workspace()
    borrowed.prepare(48); borrowed.sample(lambda:None,True); borrowed.close()
    assert odebug.counts()==(0,0,0,0,0,1)
    owner.reset_workspace(); owner.close()
    assert bdebug.counts()==(1,1,2,0,0,1)
    with pytest.raises(RuntimeError,match="closed"): borrowed.prepare(32)
    with pytest.raises(RuntimeError,match="closed"): owner.share()
