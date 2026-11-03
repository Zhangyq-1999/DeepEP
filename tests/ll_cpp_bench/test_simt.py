import os
import re
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("options", [[], ["SIMT_META"], ["SIMT_FLAG"], ["SIMT_META", "SIMT_FLAG"], ["LOCAL_FIRST"], ["LOCAL_FIRST", "BATCH_FINAL"]])
def test_simt_launch_and_workspace_fit_ub(tmp_path, options):
    """Exercise the real launch argument and layouts against the hardware budget.

    Host arithmetic stubs cannot catch SIMD accesses into SIMT's DCache region.
    The mixed kernel must request its arena at launch as well as bound offsets.
    """
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for host checks")
    text = (ROOT / "csrc/kernels/combine/low_latency.cpp").read_text(encoding="utf-8")
    start = text.index('extern "C" DEEPEP_EXPORT void combine_server_dedup_kernel_do(')
    launch = text[start:text.index("\n}", start) + 2]
    # Preserve the production launch arguments, replacing only CANN syntax.
    launch, count = re.subn(r"combine_server_dedup_kernel<<<(.*?)>>>",
                           r"RecordLaunch(\1)", launch, flags=re.S)
    assert count == 1
    source = tmp_path / "ub.cpp"
    source.write_text(r'''
#include <cassert>
#include <initializer_list>
#include "kernels/launch.hpp"
#include "kernels/combine/low_latency/launch.hpp"
#include "kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h"
using namespace CombineDedup;
unsigned launchedUb = 0;
struct Launch { template<class... T> void operator()(T...) {} };
Launch RecordLaunch(unsigned, unsigned ub, void*) { launchedUb=ub; return {}; }
''' + launch + r'''
int main() {
    deepep::EpServerDedupCombineKernelArgs args{};
    args.coreNum=56;
    combine_server_dedup_kernel_do(nullptr, &args);
#if defined(DEEPEP_LL_COMBINE_SIMT_META) || defined(DEEPEP_LL_COMBINE_SIMT_FLAG)
    // 256 KiB physical UB minus 8 KiB system and at least 32 KiB SIMT DCache.
    const unsigned limit=216U*1024U;
    assert(launchedUb>=kEpServerDedupCombineRecvUbBytes && launchedUb<=limit);
    assert(launchedUb>=kEpServerDedupCombineUbBytes);
#else
    const unsigned limit=248U*1024U;
    assert(launchedUb==0 && kEpServerDedupCombineRecvUbBytes==240U*1024U);
#endif
    assert(kEpServerDedupCombineRecvUbBytes<=limit);
    assert(kEpServerDedupCombineUbBytes==190U*1024U);
    for(unsigned bs:{1U,32U,48U,64U,72U,256U})
    for(unsigned tokens:{0U,1U,73U,512U,4096U,10000U})
    for(unsigned h:{256U,3584U,7168U,14336U})
    for(unsigned k:{1U,6U,8U,16U,32U})
    for(unsigned cores:{8U,28U,36U}) {
        const auto layout=EpCombineM45Layout(tokens,bs,h,k,cores);
        if(h==3584U && k==8U) assert(layout.inputLanes && layout.outputLanes);
        if(!layout.inputLanes) continue;
        assert(layout.totalBytes<=kEpServerDedupCombineRecvUbBytes);
        const unsigned resident=kEpServerDedupCombineResidentUbBytes;
        assert(resident+layout.inputOffset+layout.inputLanes*layout.inputBytes<=limit);
        assert(resident+layout.outputOffset+layout.outputLanes*layout.outputBytes<=limit);
        for(unsigned tasks:{0U,1U,2U,7U,8U,31U}) {
            const auto final=EpCombineM5Layout(layout,tasks);
            assert(final.inputLanes);
            for(unsigned i=0;i<final.inputLanes;++i) {
                const unsigned begin=final.inputOffsets[i],end=begin+layout.inputBytes;
                assert(resident+end<=kEpServerDedupCombineRecvUbBytes);
                for(unsigned j=0;j<layout.finalOutputLanes;++j) {
                    const unsigned out=layout.outputOffset+final.outputSlots[j]*layout.outputBytes;
                    assert(end<=out || begin>=out+layout.outputBytes);
                }
            }
        }
    }
}
''', encoding="utf-8")
    exe = tmp_path / "ub.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG",
                    *["-DDEEPEP_LL_COMBINE_" + opt + "=1" for opt in options],
                    "-I" + str(ROOT / "csrc"), str(source), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=30)

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
