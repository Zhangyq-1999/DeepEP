"""Load two real pybind modules with CPU ACL/SHMEM substitutes; share one session."""
import importlib.util
import csv
import json
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
#include "profile_samples.hpp"
#include "kernels/combine/low_latency/combine_core_layout.h"
namespace llbench {
constexpr uint64_t workspaceBytes=8ULL<<30;
struct Runtime {
    Options options; void* peers; aclrtStream stream; uint8_t* workspace; uint8_t* rdmaStage;
    uint64_t capacity,ffts; uint32_t cores,rdmaQps; void* progress;
};
}
namespace DispatchDedup {struct CommArgs {};}
namespace CombineDedup {inline unsigned EpCombineRecvCoreCount(unsigned n) {return n/2;}}
inline unsigned deepep_ll_combine_profile_enabled() {return DEEPEP_BENCH_PROFILE;}
inline unsigned deepep_ll_combine_return_options() {return TEST_VARIANT;}
inline unsigned deepep_ll_combine_data_l2() {return 0;}
inline uint64_t shmemx_get_ffts_config() {return 42;}
struct Event {int value=0; void record(void*) {}}
;
inline unsigned buffers=0,cases=0,launches=0;
struct Buffer {
    uint8_t* p; size_t bytes;
    explicit Buffer(size_t n):p(new uint8_t[n]),bytes(n) {++buffers;}
    ~Buffer() {--buffers;delete[] p;}
    void upload(const void* src,size_t n) {std::memcpy(p,src,n);}
    void download(void* dst,size_t n) {std::memcpy(dst,p,n);}
};
struct Combine {uint8_t* stageTimer=nullptr; unsigned timerMode=0,bs=0;};
struct Case {
    unsigned bs,variant=0; int dispatch=0,args=0; Combine combine;
    Case(const llbench::Options&,unsigned n,unsigned,void*):bs(n) {++cases;combine.bs=n;}
    Case(const llbench::Options&,unsigned n,unsigned,uint64_t ub,void*):bs(n) {
        if(ub!=192*1024) throw std::runtime_error("invalid upstream UB capacity");
        ++cases;
    }
    ~Case() {--cases;}
    void flipExpertSign(const llbench::Options&) {}
    double precision(bool) {return 0;}
    template<class Peers> void exportProducers(const llbench::Options&,const Peers&,unsigned,unsigned,unsigned) {}
};
inline uint8_t* local_base(const llbench::Runtime& r) {return r.workspace;}
inline DispatchDedup::CommArgs comm_args(const llbench::Runtime& r) {
    if(!r.workspace || !r.rdmaStage || r.ffts!=42 || r.rdmaQps!=56)
        throw std::runtime_error("invalid shared runtime");
    return {};
}
inline DispatchDedup::CommArgs context_for(const llbench::Runtime& r) {return comm_args(r);}
namespace platform_ascendc {
enum class CoreMemType {UB};
struct PlatformAscendCManager {
    static PlatformAscendCManager* GetInstance() {static PlatformAscendCManager p;return &p;}
    void GetCoreMemSize(CoreMemType,uint64_t& bytes) {bytes=192*1024;}
};
}
inline void ops_shmem_combine_do(void*,int*) {++launches;}
inline void dispatch_server_dedup_kernel_do_v2(void*,void*,unsigned) {}
inline void combine_clear_result_flags_kernel_do(void*,void*,uint64_t,unsigned) {}
inline void combine_server_dedup_kernel_do(void*,Combine* args) {
    ++launches;
    if(!DEEPEP_BENCH_PROFILE) {
        if(args->stageTimer) throw std::runtime_error("clean build has profile buffer");
        return;
    }
    using namespace CombineProfile;
    auto data=reinterpret_cast<uint64_t*>(args->stageTimer);
    for(unsigned c=0;c<56;++c) {
        auto r=data+c*kWords;
        if(r[Complete]) throw std::runtime_error("profile buffer not cleared before launch");
        r[Schema]=kSchema;r[Complete]=1;r[World]=8;r[Core]=c;r[Tokens]=args->bs;r[Level]=args->timerMode;
        r[Entry]=1000000000000ULL+c*100;r[FlushDone]=r[Entry]+1000+launches+c;
        r[Valid]=(1ULL<<Entry)|(1ULL<<FlushDone);
    }
}
extern "C" void ll_bench_entry_sync_do(void*,uint64_t) {}
#define DEEPEP_BENCH_SOURCE_ID "host-model"
#define DEEPEP_LL_COMBINE_ORIGINAL 0
#define LL_OPS_REV "test-upstream"
#include "bindings.hpp"
PYBIND11_MODULE(_ll_py_debug,m) {
    m.def("counts", [] { return py::make_tuple(deepep::sessionInits,deepep::sessionCloses,
                                             resets,buffers,cases,launches); });
    m.def("gather", [](py::bytes input) {
        using namespace CombineProfile;
        const auto bytes=input.cast<std::string>();
        std::string out;
        for(unsigned rank=0;rank<8;++rank) {
            std::vector<uint64_t> data(bytes.size()/8);
            std::memcpy(data.data(),bytes.data(),bytes.size());
            data[0]=12+rank%4;
            for(unsigned j=1;j<data.size();j+=kWords) data[j+Rank]=rank;
            out.append(reinterpret_cast<char*>(data.data()),bytes.size());
        }
        return py::bytes(out);
    });
}
''')
    def load(path, name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    modules=[]
    for index in range(4):
        path=tmp_path/f"variant{index}.pyd"
        subprocess.run([zig,"c++","-std=c++17","-shared","-O0","-DMS_WIN64",
                        f"-DTEST_VARIANT={index}", f"-DDEEPEP_LL_COMBINE_READY={20 if index else 12}",
                        f"-DDEEPEP_BENCH_PROFILE={int(index==2)}", "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4",
                        f"-DDEEPEP_LL_OPS_BASE={int(index==3)}",
                        "-I",str(tmp_path),"-I",str(HERE),"-I",str(HERE.parents[1]/"csrc"),
                        "-I",str(HERE.parent/"ll_cpp_bench"),
                        "-I",pybind11.get_include(),"-I",sysconfig.get_path("include"),
                        str(tmp_path/"module.cpp"),
                        str(Path(sys.base_prefix)/"libs"/f"python{sys.version_info.major}{sys.version_info.minor}.lib"),
                        "-o",str(path)],check=True,timeout=180,capture_output=True,text=True)
        modules.append((load(path,f"variant{index}._ll_py_combine"),load(path,f"variant{index}._ll_py_debug")))
    (base,bdebug),(opt,odebug),(profile,pdebug),(ops,opsdebug)=modules
    owner=base.Bench(0,8,0,["--profile","2"],123,"tcp://a:1")
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
    owner.reset_workspace()
    profiled=profile.Bench(owner.share())
    assert profiled.identity()["profile_build"] and not owner.identity()["profile_build"]
    assert not hasattr(owner,"begin_profile") and base.suite_api==profile.suite_api==3
    for bs in (32,48):
        profiled.prepare(bs)
        profiled.sample(lambda:None,True)  # warmup is not captured
        profiled.begin_profile(str(tmp_path/"records"),"hccl")
        for _ in range(3): profiled.sample(lambda:None,True)
        with pytest.raises(RuntimeError,match="sample counts differ"):
            profiled.finish_profile([1,2],pdebug.gather)
        profiled.finish_profile([30,10,20],pdebug.gather)
    profile_identity=profiled.identity()
    profiled.close()
    assert pdebug.counts()==(0,0,0,0,0,8)
    rows=list(csv.DictReader((tmp_path/"records/rank-0-profile.csv").open()))
    assert len(rows)==2*3*56 and {int(r['iteration']) for r in rows}=={0,1,2}
    global_rows=list(csv.DictReader((tmp_path/"records/diagnostic.csv").open()))
    assert {int(r['rank']) for r in global_rows}==set(range(8))
    traces=sorted((tmp_path/"records/perfetto").glob('*.json'))
    assert [p.name for p in traces]==[f'bs{bs}-{label}-i{i}.perfetto.json'
                                    for bs in (32,48) for label,i in (('p50',2),('worst',0))]
    trace=json.loads(traces[0].read_text())
    assert len([e for e in trace['traceEvents'] if e['name']=='profiled body'])==8*56
    # Exercise the actual offline reporter against native Python captures and
    # Python-written timing CSVs, without MPI or Device libraries.
    spec=importlib.util.spec_from_file_location("profile_worker",HERE/"run.py")
    worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    for bs in (32,48):
        result=worker.summarize([[30,10,20]]*8)
        result.update(tokens=bs,kind="profile",sync="hccl",identity=profile_identity,
                      config=dict(world=8,hidden=3584,topk=8,experts=384,dtype="bf16",route="random",seed=0,
                                  capacity=256,warmup=10,profile=2))
        worker.write_timing_artifacts(tmp_path/"records",result)
    reporter=tmp_path/"report.exe"
    subprocess.run([zig,"c++","-std=c++17","-O1",str(HERE.parent/"ll_cpp_bench/profile_report.cpp"),
                    "-o",str(reporter)],check=True,timeout=180,capture_output=True,text=True)
    report=subprocess.run([str(reporter),"--suite",str(tmp_path)],capture_output=True,text=True,check=True).stdout
    assert "entry_sync=hccl" in report and "profile" in report and "event_r=0 us=20.00" in report
    assert "BS=32" in report and "BS=48" in report and " S r/c=" in report and " R r/c=" in report
    owner.reset_workspace()
    upstream=ops.Bench(owner.share())
    identity=upstream.identity()
    assert identity["upstream_revision"]=="test-upstream" and identity["implementation"]=="ops-shmem-base"
    assert not identity["profile_build"] and identity["recv_cores"]==0
    upstream.prepare(32); upstream.dispatch(); upstream.clear_flags()
    upstream.sample(lambda:None,True)
    upstream.flip_inputs(); upstream.precision(); upstream.close()
    assert opsdebug.counts()==(0,0,0,0,0,1)  # borrowed session, independent upstream kernel
    result=worker.summarize([[12,13,14]]*8)
    result.update(tokens=32,kind="clean",sync="hccl",identity=identity,
                  config=dict(world=8,hidden=3584,topk=8,experts=384,dtype="bf16",route="random",seed=0,
                              capacity=256,warmup=10,profile=2))
    worker.write_timing_artifacts(tmp_path/"ops-shmem-base",result)
    report=subprocess.run([str(reporter),"--suite",str(tmp_path)],capture_output=True,text=True,check=True).stdout
    assert "upstream=test-upstream" in report and "no internal profile" in report
    # Reference-only selection must also own and close a session normally.
    standalone=ops.Bench(0,8,0,[],123,"tcp://a:1")
    standalone.prepare(32); standalone.sample(lambda:None,True); standalone.release_case(); standalone.close()
    assert opsdebug.counts()==(1,1,0,0,0,2)
    owner.reset_workspace(); owner.close()
    assert bdebug.counts()==(1,1,4,0,0,1)
    with pytest.raises(RuntimeError,match="closed"): borrowed.prepare(32)
    with pytest.raises(RuntimeError,match="closed"): owner.share()
