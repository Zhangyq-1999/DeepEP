"""Check the production producer exporter against ready/coalesce identity overlap."""
import csv
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("coalesce", [0, 1, 2])
def test_ready_identity_does_not_change_transport_mapping(tmp_path, coalesce):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG for Host C++ checks")
    main = (HERE/"main.cpp").read_text(encoding="utf-8")
    start = main.index("    template<class Peers>\n    void exportProducers")
    method = main[start:main.index("\n};", start)]
    source = r'''
#include "options.hpp"
#include "producer_map.hpp"
#include <filesystem>
#include <fstream>
#include <cassert>
constexpr int ACL_MEMCPY_DEVICE_TO_HOST=0;
int aclrtMemcpy(void* d,size_t,const void* s,size_t n,int) {std::memcpy(d,s,n);return 0;}
void check(int n,const char*) {assert(!n);}
namespace DispatchDedup {bool EpServerDedupEnabled(unsigned,unsigned,unsigned) {return true;}}
namespace CombineDedup {constexpr unsigned kEpServerDedupCombineMaxTokenNum=1000;}
namespace CombineCoalesce {constexpr unsigned kWorkers=4;}
struct FakeCase {
    unsigned bs=32;
    struct {uint8_t *sendCountsOut,*tokenTypeOut,*destinationIndexOut;} dispatch;
''' + method + r'''
};
struct Peers {
    std::vector<uint8_t> gather(const void* data,size_t n) const {
        std::vector<uint8_t> out(n*8);std::memcpy(out.data(),data,n);return out;
    }
};
int main(int argc,char** argv) {
    llbench::Options o;o.world=8;o.topk=1;o.output=argv[1];
    std::filesystem::create_directories(o.output);
    std::vector<int32_t> count(384),types(29,1),table(29*3);
    count.back()=29;
    for(unsigned row=0;row<29;++row) {table[row*3]=4;table[row*3+1]=row;}
    FakeCase c;c.dispatch={reinterpret_cast<uint8_t*>(count.data()),reinterpret_cast<uint8_t*>(types.data()),
                           reinterpret_cast<uint8_t*>(table.data())};
    const auto options=DEEPEP_BENCH_COALESCE ? (DEEPEP_BENCH_COALESCE<<9U)|1U : CombineDedup::CombineReadyOptions(20)|1U;
    assert((options>>9U)&3U); // Old exporter incorrectly inferred transport cores here.
    c.exportProducers(o,Peers{},56,28,options);
}
'''
    (tmp_path/"export.cpp").write_text(source,encoding="utf-8")
    exe = tmp_path/"export.exe"
    subprocess.run([zig,"c++","-std=c++17","-O1","-UNDEBUG",f"-DDEEPEP_BENCH_COALESCE={coalesce}",
                    "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4","-DDEEPEP_LL_COMBINE_READY=20",
                    "-I",str(HERE),str(tmp_path/"export.cpp"),"-o",str(exe)],check=True,timeout=120)
    subprocess.run([str(exe),str(tmp_path/"out")],check=True,timeout=15)
    rows=list(csv.DictReader((tmp_path/"out/producers.csv").open()))
    assert len(rows)==29 and rows[24]["core"]==("0" if coalesce else "24")
    assert {r["path"] for r in rows}=={("RDMA","RDMA-packed","RDMA-relay")[coalesce]}
