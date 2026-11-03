"""Sparse physical hosts must use RDMA across ranks 3/4, not peer GM mapping."""
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@pytest.mark.parametrize("world,host_ranks,first_device", [(8, 4, 12), (64, 16, 0)])
def test_host_transport_and_layout(tmp_path, world, host_ranks, first_device):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for CPU execution")
    source = tmp_path / "sparse.cpp"
    source.write_text(r'''
#include <cassert>
#include <vector>
#include "options.hpp"
#include "kernels/dispatch/low_latency/common/ll_count_assignment.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_layout.h"
#include "kernels/combine/low_latency/combine_layout.h"
#include "kernels/dispatch/low_latency/device/ll_transport.h"
int main() {
    using namespace DispatchDedup;
    constexpr unsigned world=TEST_WORLD, host=kLlPhysicalRanksPerHost, workers=4;
    static_assert(LlGroupForWorld(world)==host);
    std::string devices;
    for(unsigned i=0;i<host;++i) devices+=(i ? "," : "")+std::to_string(TEST_FIRST_DEVICE+i);
    char prog[]="bench", key[]="--devices";
    char* argv[]={prog,key,devices.data()};
    for(unsigned rank=0;rank<world;++rank) {
        auto o=llbench::Options::parse(3,argv,rank,world,rank%host);
        assert(o.device==TEST_FIRST_DEVICE+rank%host && o.ranksPerHost==host);
        CommArgs args{};args.rank=rank;args.rankSize=world;args.reserved0=host;
        args.reservedConfig[0]=LlRdma::kBackend;
        for(unsigned peer=0;peer<world;++peer) {
            assert(LlTransport::RemoteWrite(&args,peer)==(rank/host!=peer/host));
            if(rank/host==peer/host) assert(rank/host==peer/LlGroupForWorld(world));
        }
        std::vector<unsigned> owners(384);
        for(unsigned worker=0;worker<workers;++worker) {
            auto a=CountAssignment(rank,world,384,workers,worker,true);
            unsigned local=0,remote=0;
            for(unsigned t=worker;t<a.localDomain;t+=workers) {
                auto p=CountLocalPeer(rank,t,true);
                assert(p<world && rank/host==p/host);++local;
                for(unsigned e=p*(384/world);e<(p+1)*(384/world);++e)++owners[e];
            }
            for(unsigned t=worker;t<world-host;t+=workers) {
                auto p=CountRemotePeer(rank,world,t);
                assert(p<world && rank/host!=p/host);++remote;
                for(unsigned e=p*(384/world);e<(p+1)*(384/world);++e)++owners[e];
            }
            assert(local==a.localPeers && remote==(world-host)/workers);
        }
        for(auto n:owners)assert(n==1);
    }
    EpMemoryServerDedupLayout dispatch;
    CombineDedup::Layout combine;
    assert(BuildDispatchServerDedupLayout(world,256,3584,8,384,&dispatch,EpServerDedupInputMode::Plain16,56)==0);
    assert(DispatchDedupPacketUbFits(dispatch,256,8,384,world,56));
    assert(CombineDedup::BuildLayout(world,256,3584,8,384,combine,56));
    bool rejected=false;
    try { llbench::Options::parse(3,argv,0,host==4 ? 32 : 128,0); } catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
}
''', encoding="utf-8")
    dispatch = (ROOT / "csrc/kernels/dispatch/low_latency.cpp").read_text(encoding="utf-8")
    begin = dispatch.index('extern "C" DEEPEP_EXPORT uint32_t deepep_ll_ranks_per_group(')
    query = dispatch[begin:dispatch.index("\n}", begin) + 2]
    source.write_text(source.read_text(encoding="utf-8").replace("int main() {",
                      "#define DEEPEP_EXPORT\n" + query + "\nint main() {\n"
                      "assert(deepep_ll_ranks_per_group(TEST_WORLD)==DispatchDedup::LlGroupForWorld(TEST_WORLD));\n"), encoding="utf-8")
    exe = tmp_path / "sparse.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", f"-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST={host_ranks}",
                    f"-DTEST_WORLD={world}", f"-DTEST_FIRST_DEVICE={first_device}",
                    "-DDEEPEP_LL_RDMA=1", "-I" + str(HERE), "-I" + str(ROOT / "csrc"),
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), str(source), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=20)


def test_ep8_launcher_uses_two_nodes_and_four_processes():
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "/bin/bash"
    for node in (0, 1):
        r = subprocess.run([bash, str(HERE / "launch.sh"), "--ep", "8", "--ranks-per-host", "4",
                            "--hosts", "node0,node1", "--node-rank", str(node), "--dry-run"],
                           capture_output=True, text=True, check=True, timeout=10)
        assert f"--proxy-id {node}" in r.stdout
        assert "TIMESTAMP-UNIQUE" not in r.stdout
        if node == 0:
            assert "-ppn 4 -n 8" in r.stdout
        assert "-launcher fork" not in r.stdout


def test_sparse_count_transport_and_receive(tmp_path):
    """Run the production transport/reader model for four-rank physical hosts."""
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    # This model already exercises delayed MTE and RDMA completion, zero-count
    # rounds, repeated frame reuse and independently computed expert prefixes.
    source = (ROOT / "tests/ll_rdma/direct_count_test.cpp").read_text()
    source = source.replace("16U,32U,64U,128U,256U", "4U,8U,16U,32U")
    source = source.replace("32U,64U,128U,256U", "8U,16U,32U")
    source = source.replace("32U,64U,384U,1024U", "8U,16U,384U,1024U")
    source = source.replace("/16", "/4").replace("%16", "%4").replace("-16", "-4")
    source = source.replace("source/4*16", "source/4*4")
    source = source.replace("reserved0=16", "reserved0=4")
    source = source.replace("{0U,16U}", "{0U,4U}")
    # Concrete UB budgets below describe the original 16-rank host, not EP8.
    source = "\n".join(line for line in source.splitlines()
                       if "CountSendUbBytes(1,384,4,32" not in line)
    cpp = tmp_path / "sparse-count.cpp"
    cpp.write_text(source)
    exe = tmp_path / "sparse-count.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-UNDEBUG", "-DDEEPEP_LL_PHYSICAL_RANKS_PER_HOST=4",
                    "-I" + str(ROOT / "tests/ll_rdma/stubs"), "-I" + str(ROOT / "csrc"),
                    str(cpp), "-o", str(exe)], check=True, timeout=180)
    subprocess.run([str(exe)], check=True, timeout=90)

    counts = (ROOT / "csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_counts.h").read_text(encoding="utf-8")
    # Both send and receive must select the same frame format at world=8.
    assert "packedFrames_ = directRanks_;" in counts
    assert "epWorldSize_ > 16U" not in counts
