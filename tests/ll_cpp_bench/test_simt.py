import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

def test_simt_flag_and_compaction_models(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for host checks")
    exe = tmp_path / "simt.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG",
        "-I"+str(ROOT/"tests/ll_rdma/metadata_stubs"), "-I"+str(ROOT/"tests/ll_rdma/stubs"),
        "-I"+str(ROOT/"csrc"), "-I"+str(ROOT/"tests/ll_cpp_bench"),
        str(ROOT/"tests/ll_cpp_bench/simt_test.cpp"), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)


def test_production_metadata_batch_uses_padded_rows_and_owned_order(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for host checks")
    text = (ROOT / "csrc/kernels/combine/low_latency/combine_server_dedup_partial.h").read_text(encoding="utf-8")
    start = text.index("    __aicore__ inline void LoadTaskBatch(")
    method = text[start:text.index("\n#endif", start)]
    source = tmp_path / "batch.cpp"
    source.write_text(r'''
#include "kernel_operator.h"
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
#include "kernels/combine/low_latency/combine_profile_device.h"
#include <cassert>
#include <algorithm>
using namespace AscendC;
using namespace CombineDedup;
struct Test {
    uint32_t k_, cacheBegin_=0,cacheEnd_=0,waits=0;
    LocalTensor<int32_t> rows_,table2Ub_,table2Cache_;
    LocalTensor<uint32_t> table3Ub_,table3Cache_;
    GlobalTensor<int32_t> table2_;
    GlobalTensor<uint32_t> table3_;
    CombineProfile::View profile_;
    template<HardEvent E> void Sync(){++waits;SetFlag<E>(7);WaitFlag<E>(7);}
''' + method + r'''
};
int main(){
 for(unsigned k:{1U,2U,3U,4U,6U,8U,31U,32U})for(unsigned n:{1U,7U,8U,9U,31U,32U,47U}) {
  const unsigned total=103;
  std::vector<int32_t> rows(n),routes(total*3);
  std::vector<uint32_t> members(total*k*2);
  for(unsigned i=0;i<n;++i) rows[i]=(i*29+7)%total;
  for(unsigned i=0;i<routes.size();++i) routes[i]=i*11+1;
  for(unsigned i=0;i<members.size();++i) members[i]=i*23+7;
  std::vector<int32_t> t2(96+16,-99);
  std::vector<uint32_t> t3(32*k*2+16,0xabcdefab);
  Test t{k};t.rows_={rows.data()};t.table2Cache_={t2.data()};t.table3Cache_={t3.data()};
  t.table2_={routes.data()};t.table3_={members.data()};
  // Two groups, the second begins at a nonzero index as in the optional task plan.
  const unsigned split=n/2;
  for(unsigned group=0;group<2;++group){
   t.cacheEnd_=0;
   const unsigned begin=group?split:0,end=group?n:split;
   for(unsigned i=begin;i<end;++i){
    t.LoadTaskBatch(i,end);
    for(unsigned j=0;j<3;++j)assert(t.table2Ub_.GetValue(j)==routes[rows[i]*3+j]);
    for(unsigned j=0;j<k*2;++j)assert(t.table3Ub_.GetValue(j)==members[rows[i]*k*2+j]);
   }
  }
  assert(t.waits==(split+7)/8+(n-split+7)/8);
  for(unsigned i=96;i<t2.size();++i)assert(t2[i]==-99);
  for(unsigned i=32*k*2;i<t3.size();++i)assert(t3[i]==0xabcdefab);
 }
}
''')
    exe = tmp_path / "batch.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
        "-I"+str(ROOT/"tests/ll_rdma/metadata_stubs"), "-I"+str(ROOT/"tests/ll_rdma/stubs"),
        "-I"+str(ROOT/"csrc"), str(source), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)
