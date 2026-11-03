"""Return-order invariants and build admission; no device performance claims."""
import os
from pathlib import Path
import subprocess
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bash():
    if os.name == "nt":
        path = "C:/Program Files/Git/bin/bash.exe"
        if Path(path).is_file():
            return path
    path = shutil.which("bash")
    if not path:
        pytest.skip("Bash is required")
    return path


@pytest.mark.parametrize("enabled", [False, True])
def test_return_order_and_lane_admission(tmp_path, enabled):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU C++ checks")
    source = tmp_path / "schedule.cpp"
    source.write_text(r'''
#include <cassert>
#include <algorithm>
#include <vector>
#include "kernels/combine/low_latency/combine_return_schedule.h"
using namespace CombineDedup;
template<unsigned Group>void Check(unsigned groups,unsigned own) {
 std::vector<unsigned> expected,actual;
 for(unsigned i=0;i<groups;++i) expected.push_back(EpServerOrder::ServerDedupHalfOrder(groups,own,i));
#ifdef DEEPEP_LL_COMBINE_REMOTE_FIRST
 if(Group==16 && groups>1){
  expected.erase(std::find(expected.begin(),expected.end(),own));expected.push_back(own);
 }
#endif
 for(unsigned i=0;i<groups;++i)actual.push_back(CombineReturnServer<Group>(groups,own,i));
 assert(actual==expected);
 std::sort(actual.begin(),actual.end());
 for(unsigned i=0;i<groups;++i)assert(actual[i]==i);
 // DIRECT and M4 assign consecutive selected tasks round-robin. Changing
 // traversal must retain exact coverage and at most ceil(total/cores) tasks.
 for(unsigned cores:{1U,8U,20U,28U,36U})for(unsigned pattern:{0U,1U,2U}){
  std::vector<unsigned> counts(groups),prefix(groups+1),hits,loads(cores);
  for(unsigned g=0;g<groups;++g){counts[g]=pattern==0?0:pattern==1?(g==own?65:0):(g*13+7)%73;prefix[g+1]=prefix[g]+counts[g];}
  hits.resize(prefix.back());unsigned owner=0;
  for(unsigned g:expected){
   for(unsigned t=0;t<counts[g];++t){++hits[prefix[g]+t];++loads[(owner+t)%cores];}
   owner=(owner+counts[g])%cores;
  }
  for(unsigned n:hits)assert(n==1);
  for(unsigned n:loads)assert(n<=(prefix.back()+cores-1)/cores);
 }
}
int main(){
 for(unsigned groups=1;groups<=33;++groups)for(unsigned own=0;own<groups;++own){
  Check<2>(groups,own);Check<4>(groups,own);Check<16>(groups,own);
 }
 for(unsigned lanes=0;lanes<=8;++lanes){
#ifdef DEEPEP_LL_COMBINE_DEFER_RETURN
  assert(CombineDeferReturn(lanes)==(lanes>1));
#else
  assert(!CombineDeferReturn(lanes));
#endif
 }
}
''', encoding="utf-8")
    exe = tmp_path / "schedule.exe"
    flags = ["-DDEEPEP_LL_COMBINE_REMOTE_FIRST=1", "-DDEEPEP_LL_COMBINE_DEFER_RETURN=1"] if enabled else []
    subprocess.run([zig, "c++", "-std=c++17", "-O2", "-UNDEBUG", *flags,
                    "-I" + str(ROOT / "csrc"), str(source), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=30)


@pytest.mark.parametrize("option", ["--remote-first", "--defer-return", "--simt-meta", "--simt-flag", "--prefetch-return"])
def test_original_build_rejects_return_experiments(bash, option):
    script = ROOT / "tests/ll_cpp_bench/build.sh"
    result = subprocess.run([bash, str(script), "--variant", "base", option, "ON"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "Use --variant opt" in result.stderr
    result = subprocess.run([bash, str(script), option, "sometimes"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "must be ON or OFF" in result.stderr


@pytest.mark.parametrize("first,second,message", [
    ("--simt-flag", "--progress-probe", "cannot be combined"),
    ("--simt-flag", "--early-reduce", "cannot be combined"),
    ("--prefetch-return", "--early-issue", "different submission points"),
])
def test_conflicting_experiments_fail_before_build(bash, first, second, message):
    result = subprocess.run([bash, str(ROOT / "tests/ll_cpp_bench/build.sh"),
                             first, "ON", second, "ON"], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert message in result.stderr
