"""Exercise global Event selection and compact profile collection without NPU."""
import csv
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("world", [4, 64])
@pytest.mark.parametrize("coalesce,interleave,supply", [(False, False, False), (True, False, False), (False, True, False), (False, False, True)])
def test_global_sample_export(tmp_path, world, coalesce, interleave, supply):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU C++ checks")
    here = Path(__file__).parent
    (tmp_path / "mpi.h").write_text(r'''
#pragma once
#include <cassert>
#include <cstring>
#include "profile.hpp"
using MPI_Comm=int;
constexpr int MPI_SUCCESS=0, MPI_COMM_WORLD=1, MPI_COMM_NULL=0, MPI_ERRORS_RETURN=0;
constexpr int MPI_COMM_TYPE_SHARED=0, MPI_INFO_NULL=0, MPI_INT=4, MPI_BYTE=1, MPI_UINT64_T=8;
constexpr int MPI_MAX_ERROR_STRING=1024;
inline int MPI_Comm_set_errhandler(int,int) {return 0;}
inline int MPI_Comm_rank(int,int* p) {*p=0;return 0;}
inline int MPI_Comm_size(int comm,int* p) {*p=comm==MPI_COMM_WORLD ? TEST_WORLD : (TEST_WORLD<16 ? TEST_WORLD : 16);return 0;}
inline int MPI_Comm_split_type(int,int,int,int,int* p) {*p=2;return 0;}
inline int MPI_Comm_free(int*) {return 0;}
inline int MPI_Error_string(int,char*,int* n) {*n=0;return 0;}
inline int MPI_Bcast(void*,int,int,int,int) {return 0;}
inline int MPI_Barrier(int) {return 0;}
inline int MPI_Allgather(const void* p,int n,int type,void* out,int,int,int) {
    using namespace CombineProfile;
    assert(type==MPI_BYTE && (n==(1+9*kWords)*sizeof(uint64_t) || n==(1+6*kWords)*sizeof(uint64_t)));
    const uint32_t rows=(n/sizeof(uint64_t)-1)/kWords;
    for (uint32_t rank=0;rank<TEST_WORLD;++rank) {
        auto dest=static_cast<uint8_t*>(out)+rank*n;
        std::memcpy(dest,p,n);
        const uint64_t device=TEST_WORLD==4 ? rank+8 : rank%16;
        std::memcpy(dest,&device,sizeof(device));
        for (uint32_t j=0;j<rows;++j) {
            const uint64_t r=rank;
            std::memcpy(dest+(1+j*kWords+Rank)*8,&r,sizeof(r));
        }
    }
    return 0;
}
''')
    (tmp_path / "test.cpp").write_text(r'''
#include "profile_samples.hpp"
#include "mpi_coordinator.hpp"
#include "sample_report.hpp"
#include <cassert>
int main(int argc,char** argv) {
    using namespace CombineProfile;
    assert(argc==2);
    llbench::Options o;
    o.world=TEST_WORLD; o.iterations=3; o.output=argv[1]; o.profile=2;
    std::vector<float> samples(TEST_WORLD*3, 1.0f);
    const float first[12]={10,30,13, 11,29,12, 9,28,12, 10,27,12};
    std::memcpy(samples.data(),first,sizeof(first));
    if (TEST_WORLD==64) samples[63*3+2]=40;
    std::vector<uint8_t> data(samples.size()*sizeof(float));
    std::memcpy(data.data(),samples.data(),data.size());
    const auto maxima=llbench::rank_max_samples(data,o,32);
    assert((maxima==std::vector<float>{11,30,TEST_WORLD==64 ? 40.0f : 13.0f}));
    bool rejected=false;
    data.pop_back();
    try {llbench::rank_max_samples(data,o,32);} catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
    llbench::ProfileSamples profiles;
    for (uint32_t iteration=0;iteration<3;++iteration) {
        std::vector<uint64_t> raw(6*kWords);
        for (uint32_t core=0;core<6;++core) {
            auto row=raw.data()+core*kWords;
            row[Schema]=kSchema; row[Complete]=1; row[Core]=core;
            row[Rank]=0; row[World]=TEST_WORLD; row[Tokens]=32; row[Level]=2;
            row[Entry]=(uint64_t(1)<<60)+iteration*10000;
            row[FlushDone]=row[Entry]+100+core*10;
            row[Valid]=(uint64_t(1)<<Entry)|(uint64_t(1)<<FlushDone);
            row[StageRows]=iteration*10+core;
        }
        // Core 2 is longest sender, 5 longest receiver; core 1 has latest Put.
        auto late=raw.data()+kWords;
        late[RdmaLastPut]=late[Entry]+90;
        late[Valid]|=uint64_t(1)<<RdmaLastPut;
        if (TEST_COALESCE) {
            raw[CoalesceWorker]=1;
            raw[FlushDone]=raw[Entry]+250;
            late[CoalesceWorker]=1;
        }
        if (TEST_SUPPLY) {
            auto stamp=[&](unsigned core,Word w,unsigned ticks) {
                auto r=raw.data()+core*kWords;r[w]=r[Entry]+ticks;r[Valid]|=1ULL<<w;
            };
            // Fast bodies that would be absent from the old four selections.
            stamp(0,StageEnd,60);stamp(3,StageDrained,80);
            stamp(4,NotifyBegin,20);stamp(4,NotifyLocalReady,30);stamp(4,NotifyEnd,90);
            stamp(3,ReadyBegin,20);stamp(3,ReadyObserved,100);
            raw[4*kWords+PartialLocalWaitTicks]=20;raw[4*kWords+PartialGetWaitTicks]=30;
        }
        profiles.capture(raw,6,3);
    }
    llbench::Coordinator peers;
    profiles.write(peers,o,32,6,3,maxima);
}
''')
    exe = tmp_path / "export.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
                    "-DTEST_WORLD=" + str(world),
                    "-DTEST_COALESCE=" + str(int(coalesce)),
                    "-DTEST_SUPPLY=" + str(int(supply)),
                    "-DDEEPEP_LL_COMBINE_READY=" + str(18 if interleave else 0),
                    "-I" + str(tmp_path), "-I" + str(here), str(tmp_path / "test.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    subprocess.run([str(exe), str(tmp_path)], check=True, timeout=30)
    with (tmp_path / "rank-max.csv").open() as stream:
        events = list(csv.DictReader(stream))
    assert [(int(r["iteration"]), int(r["rank"]), float(r["us"])) for r in events] == [
        (0, 1, 11), (1, 0, 30), (2, 63, 40) if world == 64 else (2, 0, 13)]
    with (tmp_path / "diagnostic.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 3 * world * (6 if supply else 4 if coalesce else 3)
    assert {int(r["rank"]) for r in records} == set(range(world))
    assert {int(r["iteration"]) for r in records} == {0, 1, 2}
    for row in records:
        core = int(row["core"])
        assert core in (tuple(range(6)) if supply else (0, 1, 2, 5) if coalesce else (1, 4, 5) if interleave else (1, 2, 5))
        assert row["role"] == ("receive" if (core % 2 == 0 if interleave else core >= 3) else "send")
        rank = int(row["rank"])
        assert int(row["device"]) == (rank + 8 if world == 4 else rank % 16)
        assert int(row["stage_rows"]) == int(row["iteration"]) * 10 + int(row["core"])
        assert int(row["entry_tick"]) == (1 << 60) + int(row["iteration"]) * 10000
    traces = list((tmp_path / "perfetto").glob("*.json"))
    assert len(traces) == 2
    for path in traces:
        trace = json.loads(path.read_text())
        bodies = [e for e in trace["traceEvents"] if e["name"] == "profiled body"]
        assert len(bodies) == world * 6
        assert {e["args"]["rank"] for e in bodies} == set(range(world))
        iteration = trace["otherData"]["iteration"]
        assert iteration in (1, 2)
        assert all(e["args"]["stage_rows"] == iteration * 10 + e["args"]["core"] for e in bodies)
        labels = [e["args"]["name"] for e in trace["traceEvents"] if e["name"] == "thread_name"]
        if interleave:
            for core in range(6):
                role = "RECEIVE" if core % 2 == 0 else "SEND"
                assert any(label.startswith(f"AIV {core} {role} / stages") for label in labels)
