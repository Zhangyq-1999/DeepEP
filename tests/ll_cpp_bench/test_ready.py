"""Run production readiness checks against asynchronous CPU participants."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
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


@pytest.mark.parametrize("mode,pipeline", [(1, False), (3, False), (3, True), (4, False), (5, False), (6, False), (7, False), (8, False), (9, False), (10, False), (11, False), (12, False), (13, False), (14, False), (15, False), (16, False), (17, False), (18, False), (19, False), (20, False), (21, False), (22, False), (23, False), (24, False), (25, False), (26, False), (27, False), (28, False), (29, False), (30, False)])
def test_production_dependency_scheduler(tmp_path, mode, pipeline):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the production scheduler check")
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    start = text.rindex("#if", 0, text.index("    __aicore__ inline uint32_t ProcessReadyTasks("))
    end = text.index("    template <class T>\n    __aicore__ inline LocalTensor<T> Allocate", start)
    method = text[start:end]
    preload = text[text.index("    __aicore__ inline void PreloadInputs("):text.index("    template <class T>\n    __aicore__ inline void Aggregate(")]
    preload = preload.split("#if DEEPEP_LL_COMBINE_READY == 14")[0]
    process = text[text.index("    __aicore__ inline uint32_t Process()\n"):text.index("\nprivate:", text.index("    __aicore__ inline uint32_t Process()\n"))]
    load = text[text.index("    __aicore__ inline void LoadTask(uint32_t row)"):text.index("#ifdef DEEPEP_LL_COMBINE_SIMT_META", text.index("    __aicore__ inline void LoadTask(uint32_t row)"))]
    load = load.replace("        // Route", "        ++loads;\n        // Route")
    cpp, exe = tmp_path / "scheduler.cpp", tmp_path / "scheduler.exe"
    cpp.write_text(r'''
#define main ReadyProtocolChecks
#include "ready_test.cpp"
#undef main
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
#include "kernels/combine/low_latency/combine_return_schedule.h"
#include "kernels/combine/low_latency/combine_partial_plan.h"
#include "kernels/dispatch/low_latency/common/dispatch_dedup_types.h"
struct half{};
struct bfloat16_t{};

unsigned filterCalls=0;
uint32_t CountSelect(LocalTensor<int32_t> d,LocalTensor<int32_t> w,LocalTensor<int32_t> t,
                     LocalTensor<uint8_t> m,uint32_t n,uint32_t missing,bool remote) {
    ++filterCalls;return SelectReadyCombineTasks(d,w,t,m,n,missing,remote);
}
#define SelectReadyCombineTasks CountSelect
template<uint32_t Group> struct Scheduler {
    struct Task {uint32_t mask,source;};
    struct Ready {
        uint32_t round=0,mask=0,delay=1;
        std::vector<uint32_t> sequence;
        const bool* pending=nullptr;
        uint32_t Poll() {
#if DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
            assert(pending && !*pending); // No unposted packet across ready polling.
#endif
            mask=sequence.empty() ? (round++ >= delay ? (1U<<Group)-1U : 1U) : sequence[std::min<size_t>(round++,sequence.size()-1)];return mask;
        }
        bool All(){return mask==((1U<<Group)-1U);}
    } ready;
    struct Batch {
        bool pending=false;uint32_t lane=0,posted=0;
        void Init(DispatchDedup::CommArgs*,uint32_t){}
        uint32_t staged=0, deferredAt=0;
        uint32_t* aggregateCount=nullptr;
        bool deferred=false;
        bool HasStaged()const{return pending;}
        void SubmitPending(){
#if DEEPEP_LL_COMBINE_READY == 22 || DEEPEP_LL_COMBINE_READY == 23
            if(pending && deferred)assert(*aggregateCount==deferredAt+1U);
#endif
            if(pending){posted+=staged;staged=0;pending=false;deferred=false;}
        }
    } batch;
    Batch* rdmaBatch_=&batch;
    Ready* ready_=&ready;
    CombineProfile::View profile_;
    DispatchDedup::CommArgs args{};
    DispatchDedup::CommArgs* args_=&args;
    uint32_t k_=8,localExperts_=1,outputLanes_=3,dtype_=0,loads=0,overlaps=0;
    uint32_t aggregated=0,inputSequence_=0,earlyReads=0,usedReads=0;
    uint32_t prefetchTask_=0xffffffffU,prefetchMask_=0U;
    uint64_t returnOffset_=0U,window_=0U;
    std::vector<Task> input;
    std::vector<uint32_t> sent;
    std::vector<int32_t> gm2;
    std::vector<uint32_t> gm3;
    std::array<int32_t,128> rows{},deps{},work{},selected{};
    std::array<int32_t,96+8> table2{};
    std::array<uint32_t,2048+8> table3{};
    alignas(8) std::array<uint8_t,32> mask{};
    LocalTensor<int32_t> rows_{rows.data()},readyDeps_{deps.data()},readyWork_{work.data()},readyTasks_{selected.data()},table2Ub_{table2.data()};
    LocalTensor<uint32_t> table3Ub_{table3.data()};
    LocalTensor<uint8_t> readyMask_{mask.data()};
    GlobalTensor<int32_t> table2_;
    GlobalTensor<uint32_t> table3_;
    LocalTensor<int32_t> readyTable2_;
    LocalTensor<uint32_t> readyTable3_;
    uint32_t readyTable3Stride_=0;
    bool readyCached_=false;
#if DEEPEP_LL_COMBINE_READY >= 4
    PartialTaskPlan<Group> plan_;
    bool planned_=false;
    std::vector<int32_t> types;
    GlobalTensor<int32_t> types_;
    std::vector<uint64_t> arena;
    LocalTensor<uint8_t> planScratch_,outputs_;
    uint32_t planScratchBytes_=0,tokens_=0,recvCores_=1,inputLanes_=kEpServerDedupCombineM4MaxInputLanes,blocks_=15,packetBytes_=7680;
#endif
    explicit Scheduler(std::vector<Task> tasks,uint32_t world=2*Group,uint32_t rank=0,uint32_t topk=8):k_(topk),input(std::move(tasks)) {
        batch.aggregateCount=&aggregated;
        ready.pending=&batch.pending;
        args.rank=rank;args.rankSize=world;args.reserved0=Group;args.reservedConfig[0]=LlRdma::kBackend;
        dtype_=DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
#if DEEPEP_LL_COMBINE_READY >= 4
        tokens_=input.size();types.resize(tokens_,2);types_.SetGlobalBuffer(types.data());
        const auto layout=EpCombineM45Layout(tokens_,72,3584,k_,recvCores_);
        assert(layout.outputLanes==8U);
        arena.resize((kEpServerDedupCombineRecvUbBytes+64)/8,0x5b5b5b5b5b5b5b5bULL);
        auto scratch=reinterpret_cast<uint8_t*>(arena.data())+kEpServerDedupCombineResidentUbBytes;
        planScratch_={scratch+layout.inputOffset};planScratchBytes_=EpCombineRouteSnapshotBytes(layout);
        outputs_={scratch+layout.outputOffset};
#endif
        gm2.resize(input.size()*3U);gm3.resize(input.size()*k_*2U);
        table2_.SetGlobalBuffer(gm2.data());table3_.SetGlobalBuffer(gm3.data());
        std::fill(table2.begin(),table2.end(),-123);std::fill(table3.begin(),table3.end(),0xcafebabeU);
        for(uint32_t row=0;row<input.size();++row) {
            gm2[row*3U]=input[row].source;gm2[row*3U+1U]=row;gm2[row*3U+2U]=row%k_;
            uint32_t n=0;
            for(uint32_t local=0;local<Group;++local) if(input[row].mask&(1U<<local))
                gm3[row*k_*2U+2U*(++n)]=(args.rank/Group*Group+local)|((row%32U)<<16U);
            gm3[row*k_*2U]=n;
            for(uint32_t member=0;member<=n;++member) gm3[row*k_*2U+member*2U+1U]=0x3f000000U+row*32U+member;
        }
    }
    template<HardEvent E>void Sync(){SetFlag<E>(7);WaitFlag<E>(7);}
    uint32_t SelectServer(uint32_t server,uint32_t&,bool,uint32_t append) {
        for(uint32_t row=0;row<input.size();++row) if(input[row].source/Group==server) rows[append++]=row;
        return append;
    }
#if DEEPEP_LL_COMBINE_READY >= 4
    void Load(uint32_t row,uint32_t source,uint32_t member) {
        assert(source==input[row].source);
        if(member) {
            const uint32_t rank=(table3Ub_.GetValue(member*2U)&0xffffU)/localExperts_;
            assert(ready.mask&(1U<<(rank%Group)));
        }
        const uint32_t lane=(inputSequence_+member)%inputLanes_;
        WaitFlag<HardEvent::V_MTE2>(lane);
        SetFlag<HardEvent::MTE2_V>(lane);
        if(input[row].mask & ~ready.mask) ++earlyReads;
    }
#endif
    template<class T>void Aggregate(uint32_t row,uint32_t source,uint32_t members) {
#if DEEPEP_LL_COMBINE_READY >= 4
        // The route snapshot aliases the output ring. Process must initialize
        // all data-with-flag tails AFTER planning, including untouched lanes.
        for(uint32_t lane=0;lane<(kCombineLeanSetup?std::min<uint32_t>(input.size(),outputLanes_):outputLanes_);++lane)for(uint32_t block=0;block<blocks_;++block)
            for(uint32_t j=0;j<8;++j)assert(outputs_[lane*packetBytes_+block*512U+480U].template ReinterpretCast<float>().GetValue(j)==1.f);
#endif
        assert(source==input[row].source && members==table3Ub_.GetValue(0U)+1U);
        assert(!std::memcmp(table2Ub_.p,gm2.data()+row*3U,12U));
        assert(!std::memcmp(table3Ub_.p,gm3.data()+row*k_*2U,k_*8U));
        assert(!(input[row].mask & ~ready.mask)); // Arithmetic only after all members ready.
#if DEEPEP_LL_COMBINE_READY >= 4
        usedReads+=__builtin_popcount(prefetchMask_);
        PreloadInputs(row,source,members);
        for(uint32_t member=0;member<members;++member) {
            const uint32_t lane=(inputSequence_+member)%inputLanes_;
            WaitFlag<HardEvent::MTE2_V>(lane);
            SetFlag<HardEvent::V_MTE2>(lane);
#if DEEPEP_LL_COMBINE_READY == 17
            if(member==0) {
                for(uint32_t next=1;next<=std::min(members-1,inputLanes_);++next)Load(row,source,next);
            } else if(member+inputLanes_<members)Load(row,source,member+inputLanes_);
#else
            if(member+inputLanes_<members)Load(row,source,member+inputLanes_);
#endif
        }
        inputSequence_+=members;
#endif
        for(uint32_t previous:sent) assert(previous!=row);
        if(batch.pending) {assert(batch.lane!=sent.size()%outputLanes_);++overlaps;}
#if DEEPEP_LL_COMBINE_READY == 6 || DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 21
        batch.SubmitPending(); // Production Aggregate submits after input preload.
#endif
        ++aggregated;
    }
    void Send(uint32_t lane,uint32_t source,uint32_t token,uint32_t topk,bool defer) {
        assert(lane==sent.size()%outputLanes_ && source==input[token].source && topk==token%k_);
#if DEEPEP_LL_COMBINE_READY == 7
        batch.SubmitPending(); // Send queues the current Vector packet first.
#endif
#if DEEPEP_LL_COMBINE_READY != 5
        assert(!batch.pending); // Ordinary batch has one unposted packet.
#endif
#if DEEPEP_LL_COMBINE_READY == 7
        assert(returnOffset_==COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET+(uint64_t(token)*k_+topk)*packetBytes_);
#endif
        sent.push_back(token);
        if(LlTransport::RemoteWrite(args_,source)) {
            batch.pending=true;batch.lane=lane;++batch.staged;
            batch.deferred=defer;batch.deferredAt=aggregated;
#if DEEPEP_LL_COMBINE_READY == 5
            if(batch.staged==2)batch.SubmitPending();
#endif
            if(!defer)batch.SubmitPending();
        } else batch.SubmitPending();
    }
''' + load + method + ('\n' + preload if mode >= 4 else '') + ('\n' + process if mode >= 4 else '') + r'''
    uint32_t Run(uint32_t servers,uint32_t own) {
#if DEEPEP_LL_COMBINE_READY >= 4
        cores=2;core=CombineDedup::kCombineInterleavedCores?0:1; // One receiver owns every FIRST_HOP in this fixture.
        for(uint32_t lane=0;lane<outputLanes_;++lane)SetFlag<HardEvent::MTE3_V>(lane);
        const auto result=Process();
        for(uint32_t lane=0;lane<outputLanes_;++lane)WaitFlag<HardEvent::MTE3_V>(lane);
        for(uint32_t i=kEpServerDedupCombineRecvUbBytes;i<arena.size()*8U;++i)
            assert(reinterpret_cast<uint8_t*>(arena.data())[i]==0x5b);
        return result;
#else
        return ProcessReadyTasks(servers,own);
#endif
    }
};
template<uint32_t Group>void Check() {
    // The first remote task depends on the delayed rank. A ready local task
    // must still run; when both become ready, the remote task goes first.
    Scheduler<Group> t({{2U,Group},{1U,0U},{2U,0U}});
    assert(t.Run(2,0)==3U);
    assert((t.sent==std::vector<uint32_t>{1U,0U,2U}));
    assert(t.aggregated==3U);
    Scheduler<Group> empty({});
    assert(empty.Run(2,0)==0 && empty.ready.round==0);
    Scheduler<Group> invalid({{0U,Group}});
    bool trapped=false;try{invalid.Run(2,0);}catch(const std::runtime_error&){trapped=true;}
    assert(trapped);
#if DEEPEP_LL_COMBINE_READY >= 4
    // Trap aborts the real kernel; discard the fixture's seeded payload events.
    for(auto& direction:events)for(auto& live:direction)live=false;
#endif
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
        assert(t.Run(4,own)==5U);
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

void CacheAndPipeline() {
#if DEEPEP_LL_COMBINE_READY >= 4
    Scheduler<16> fallback({{2U,16U},{2U,32U},{2U,48U}},64,0);
    fallback.planScratchBytes_=0;
    assert(fallback.Run(4,0)==3U && !fallback.planned_);
    Scheduler<16> snapshot({{2U,16U},{2U,32U},{2U,48U}},64,0);
    assert(snapshot.Run(4,0)==3U && snapshot.planned_);
    assert(fallback.sent==snapshot.sent);
    NoLiveEvents();
#endif
    for(uint32_t k:{2U,3U,8U,32U}) for(uint32_t count:{1U,2U,3U,8U,12U,13U,65U}) for(uint32_t lanes:{1U,2U,8U}) {
        std::vector<Scheduler<16>::Task> tasks;
        for(uint32_t i=0;i<count;++i) tasks.push_back({2U,((i%3U)+1U)*16U+15U});
        Scheduler<16> t(tasks,64,0,k);t.outputLanes_=lanes;t.ready.delay=5;
        const auto before=filterCalls;
        assert(t.Run(4,0)==count);
        assert(t.aggregated==count && !t.batch.pending && t.batch.posted==count);
#if DEEPEP_LL_COMBINE_READY >= 3
        assert(filterCalls-before==(kCombineSingleTask && count==1 ? 0U : 3U+(DEEPEP_LL_COMBINE_READY==13 && t.readyCached_ ? 1U:0U)));
        assert(t.loads==(count<=EpCombineReadyCacheCapacity(k)?0U:count*2U));
#if defined(DEEPEP_LL_COMBINE_DEFER_RETURN) || DEEPEP_LL_COMBINE_READY == 6 || (DEEPEP_LL_COMBINE_READY == 11 || DEEPEP_LL_COMBINE_READY == 15 || DEEPEP_LL_COMBINE_READY == 23)
        assert(t.overlaps==(lanes>1U?count-1U:0U));
#elif DEEPEP_LL_COMBINE_READY == 7 || DEEPEP_LL_COMBINE_READY == 21 || DEEPEP_LL_COMBINE_READY == 22
        assert(t.overlaps==(lanes>1U && count>2U?count-2U:0U));
#endif
#else
        assert(filterCalls-before==11U && t.loads==count*2U);
#endif
        // Check immediately beyond the actual reserved caches, including K
        // values whose per-row padded stride differs from their GM stride.
        const uint32_t table2End=EpCombineAlign(kEpServerDedupCombineM4TableRows*12U,32U)/4U;
        const uint32_t table3End=EpCombineAlign(kEpServerDedupCombineM4TableRows*k*8U,32U)/4U;
        for(uint32_t i=table2End;i<t.table2.size();++i)assert(t.table2[i]==-123);
        for(uint32_t i=table3End;i<t.table3.size();++i)assert(t.table3[i]==0xcafebabeU);
        NoLiveEvents();
    }
}
void EarlyInputs() {
#if DEEPEP_LL_COMBINE_READY == 13
    // Partially ready reads are reused when that task wins.
    Scheduler<16> reuse({{6U,16U}},64);
    reuse.ready.sequence={0U,2U,2U,6U,65535U};
    assert(reuse.Run(4,0)==1 && reuse.earlyReads==2 && reuse.usedReads==2);
    NoLiveEvents();
    // A different remote task becomes runnable first: discard the speculative
    // reads, let it run immediately, then reload the old task after ring advance.
    Scheduler<16> cancel({{8U,16U},{2U,32U}},64);
    cancel.ready.sequence={0U,2U,10U,65535U};
    assert(cancel.Run(4,0)==2 && cancel.earlyReads==1 && cancel.usedReads==0);
    assert((cancel.sent==std::vector<uint32_t>{1U,0U}));
    NoLiveEvents();
    // More members than lanes: refill and wrap without reordering contributions.
    Scheduler<16> wrap({{127U,16U},{127U,32U}},64,0,32);
    wrap.ready.sequence={3U,15U,127U,65535U};
    assert(wrap.Run(4,0)==2 && wrap.usedReads==5 && wrap.inputSequence_==16);
    NoLiveEvents();
#endif
}
int main(){Check<2>();Check<4>();Check<16>();FourHosts();CacheAndPipeline();EarlyInputs();}

''', encoding="utf-8")
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_TEST_DEFERRED_MTE2=1", f"-DDEEPEP_LL_COMBINE_READY={mode}",
                    *(["-DDEEPEP_LL_COMBINE_DEFER_RETURN=1"] if pipeline else []),
                    "-I" + str(ROOT / "tests/ll_cpp_bench"), "-I" + str(ROOT / "tests/ll_rdma/metadata_stubs"),
                    "-I" + str(ROOT / "csrc"), str(cpp), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)
