"""Run production readiness checks against asynchronous CPU participants."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", [1, 2])
def test_ready_lifecycle_and_vector_selection(tmp_path, mode):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the production protocol check")
    exe = tmp_path / "ready.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    f"-DDEEPEP_LL_COMBINE_READY={mode}",
                    "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "csrc"),
                    str(ROOT / "tests/ll_cpp_bench/ready_test.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_production_dependency_scheduler(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the production scheduler check")
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    start = text.index("    __aicore__ inline uint32_t ProcessReadyTasks(")
    end = text.index("\n#endif", start)
    method = text[start:end]
    cpp, exe = tmp_path / "scheduler.cpp", tmp_path / "scheduler.exe"
    cpp.write_text(r'''
#define main ReadyProtocolChecks
#include "ready_test.cpp"
#undef main
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
#include "kernels/combine/low_latency/combine_return_schedule.h"
#include "kernels/dispatch/low_latency/common/dispatch_dedup_types.h"
struct half{};
struct bfloat16_t{};
template<uint32_t Group> struct Scheduler {
    struct Task {uint32_t mask,source;};
    struct Ready {
        uint32_t round=0,mask=0;
        uint32_t Poll() {mask=round++ ? (1U<<Group)-1U : 1U;return mask;}
        bool All(){return mask==((1U<<Group)-1U);}
    } ready;
    Ready* ready_=&ready;
    CombineProfile::View profile_;
    DispatchDedup::CommArgs args{};
    DispatchDedup::CommArgs* args_=&args;
    uint32_t k_=8,localExperts_=1,outputLanes_=3,dtype_=0;
    uint32_t current=0,aggregated=0;
    std::vector<Task> input;
    std::vector<uint32_t> sent;
    std::array<int32_t,128> rows{},deps{},work{},selected{},table2{};
    std::array<uint32_t,16> table3{};
    alignas(8) std::array<uint8_t,32> mask{};
    LocalTensor<int32_t> rows_{rows.data()},readyDeps_{deps.data()},readyWork_{work.data()},readyTasks_{selected.data()},table2Ub_{table2.data()};
    LocalTensor<uint32_t> table3Ub_{table3.data()};
    LocalTensor<uint8_t> readyMask_{mask.data()};
    explicit Scheduler(std::vector<Task> tasks,uint32_t world=2*Group,uint32_t rank=0):input(std::move(tasks)) {
        args.rank=rank;args.rankSize=world;args.reserved0=Group;args.reservedConfig[0]=LlRdma::kBackend;
        dtype_=DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    }
    template<HardEvent E>void Sync(){SetFlag<E>(7);WaitFlag<E>(7);}
    uint32_t SelectServer(uint32_t server,uint32_t&,bool,uint32_t append) {
        for(uint32_t row=0;row<input.size();++row) if(input[row].source/Group==server) rows[append++]=row;
        return append;
    }
    void LoadTask(uint32_t row) {
        current=row;table2[0]=input[row].source;table2[1]=row;table2[2]=0;
        uint32_t n=0;
        for(uint32_t rank=0;rank<Group;++rank) if(input[row].mask&(1U<<rank)) table3[2U*(++n)]=args.rank/Group*Group+rank;
        table3[0]=n;
    }
    template<class T>void Aggregate(uint32_t row,uint32_t source,uint32_t members) {
        assert(row==current && source==input[row].source && members==table3[0]+1U);
        assert(!(input[row].mask & ~ready.mask)); // No unready remote stage read.
        for(uint32_t previous:sent) assert(previous!=row);
        ++aggregated;
    }
    void Send(uint32_t lane,uint32_t source,uint32_t token,uint32_t topk,bool defer) {
        assert(lane==sent.size()%outputLanes_ && source==input[current].source && token==current && topk==0 && !defer);
        sent.push_back(current);
    }
''' + method + r'''
};
template<uint32_t Group>void Check() {
    // The first remote task depends on the delayed rank. A ready local task
    // must still run; when both become ready, the remote task goes first.
    Scheduler<Group> t({{2U,Group},{1U,0U},{2U,0U}});
    assert(t.ProcessReadyTasks(2,0)==3U);
    assert((t.sent==std::vector<uint32_t>{1U,0U,2U}));
    assert(t.aggregated==3U);
    Scheduler<Group> empty({});
    assert(empty.ProcessReadyTasks(2,0)==0 && empty.ready.round==0);
    Scheduler<Group> invalid({{0U,Group}});
    bool trapped=false;try{invalid.ProcessReadyTasks(2,0);}catch(const std::runtime_error&){trapped=true;}
    assert(trapped);
    NoLiveEvents();
}
void FourHosts() {
    for(uint32_t rank:{0U,15U,16U,31U,32U,47U,48U,63U}) {
        const uint32_t own=rank/16U;
        // The local task can run before a dependency on local rank 15.
        // Once that dependency arrives, return to all THREE remote servers
        // before the remaining local task; no fixed peer/server pair.
        std::vector<Scheduler<16>::Task> tasks{{1U,rank},{0x8001U,rank}};
        for(uint32_t server=0;server<4;++server) if(server!=own)
            tasks.push_back({0x8001U,server*16U+15U});
        Scheduler<16> t(tasks,64,rank);
        assert(t.ProcessReadyTasks(4,own)==5U);
        assert(t.sent.front()==0U && t.sent.back()==1U && t.aggregated==5U);
        uint32_t visited=0;
        for(uint32_t i=1;i<4;++i) {
            const uint32_t server=tasks[t.sent[i]].source/16U;
            assert(server!=own && !(visited&(1U<<server)));visited|=1U<<server;
        }
        assert(visited==(15U^(1U<<own)));
        NoLiveEvents();
    }
}
int main(){Check<2>();Check<4>();Check<16>();FourHosts();}
''', encoding="utf-8")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_LL_COMBINE_READY=1",
                    "-I" + str(ROOT / "tests/ll_cpp_bench"), "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)
