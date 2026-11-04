"""Host checks for the actual Python launcher, sampling and native gate ordering."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("ll_python_"+name,HERE/(name+".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_default_and_four_host_plan():
    launch = module("launch")
    args = launch.parser().parse_args(["--node-rank","0"])
    node, hosts, run_id, out, build, cmd = launch.plan(args)
    assert node==0 and len(hosts)==2 and build.name=="ll-cpp-base"
    assert cmd[cmd.index("--tokens")+1]=="32,48,64,72,256"
    assert cmd[cmd.index("--devices")+1]=="12,13,14,15"
    assert "--nproc_per_node=4" in cmd
    assert cmd[cmd.index("--sync")+1]=="device"
    assert launch.plan(args)[2]!=run_id
    other = launch.parser().parse_args([run_id,"--node-rank","1"])
    assert launch.plan(other)[3]==out
    devices = ",".join(map(str,range(16)))
    args = launch.parser().parse_args(["same-run","--hosts","a,b,c,d","--node-rank","3",
                                      "--ranks-per-host","16","--devices",devices])
    cmd = launch.plan(args)[-1]
    assert "--nnodes=4" in cmd and "--nproc_per_node=16" in cmd


@pytest.mark.parametrize("extra",[["--devices","12,12,14,15"],["--node-rank","2"],["--tokens","32","--iters","0"],
                                 ["--master-port","65535"],["--hosts","a,a"]])
def test_invalid_launch(extra):
    launch = module("launch")
    with pytest.raises(ValueError): launch.plan(launch.parser().parse_args(["--node-rank","0",*extra]))


def test_rank_max_then_percentile_and_full_samples():
    run = module("run")
    result = run.summarize([[10,100,12],[20,30,80]])
    assert result["rank_max_us"]==[20,100,80]
    assert result["rank_max"]["p50"]==80
    assert result["rank0"]["p50"]==12
    data = [[1.0]*50+[2.0]*50,[3.0]*100]
    assert run.summarize(data)["rank_max_tail50"]["n"]==50
    for bad in ([[1],[1,2]],[[1,float("nan")]],[[0]],[]):
        with pytest.raises(ValueError): run.summarize(bad)


def test_worker_environment_disables_blocking_wait(tmp_path,monkeypatch):
    launch=module("launch")
    for key in ("TORCH_HCCL_BLOCKING_WAIT","HCCL_BLOCKING_WAIT","ASCEND_VISIBLE_DEVICES"):
        monkeypatch.setenv(key,"1")
    env=launch.worker_environment(launch.parser().parse_args([]),tmp_path)
    assert env["TORCH_HCCL_BLOCKING_WAIT"]==env["HCCL_BLOCKING_WAIT"]=="0"
    assert env["ASCEND_RT_VISIBLE_DEVICES"]=="12,13,14,15"
    assert "ASCEND_VISIBLE_DEVICES" not in env


def test_device_gate_drains_host_queue_after_work_wait():
    run = module("run")
    calls = []
    class Stream:
        @property
        def npu_stream(self): calls.append("drain"); return 7
    torch = SimpleNamespace(float32=0,ones=lambda *a,**kw:object(),
        npu=SimpleNamespace(synchronize=lambda:calls.append("sync"),current_stream=Stream))
    def all_reduce(*args,**kwargs):
        assert kwargs["async_op"] and kwargs["op"]=="max"
        calls.append("reduce")
        return SimpleNamespace(wait=lambda:calls.append("wait"))
    dist = SimpleNamespace(new_group=lambda **k:1,all_reduce=all_reduce,ReduceOp=SimpleNamespace(MAX="max"))
    gate = run.DeviceGate(torch,dist,180)
    assert calls==["reduce","wait","drain","sync"]
    calls.clear(); gate()
    assert calls==["reduce","wait","drain"]


def test_native_gate_queues_everything_before_release(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig: pytest.skip("Set DEEPEP_TEST_ZIG")
    (tmp_path/"acl").mkdir()
    (tmp_path/"acl/acl.h").write_text(r'''
#pragma once
#include <cstdint>
#include <vector>
inline std::vector<int> calls;
using aclrtStream=void*; using aclrtNotify=void*;
inline int aclrtCreateNotify(void** p,int) {*p=(void*)1;return 0;}
inline int aclrtCreateStream(void** p) {*p=(void*)2;return 0;}
inline int aclrtDestroyNotify(void*) {calls.push_back(8);return 0;}
inline int aclrtDestroyStream(void*) {calls.push_back(7);return 0;}
inline int aclrtWaitAndResetNotify(void*,void*,unsigned) {calls.push_back(1);return 0;}
inline int aclrtRecordNotify(void*,void*) {calls.push_back(6);return 0;}
inline int aclrtSynchronizeStreamWithTimeout(void*,int) {return 0;}
''')
    (tmp_path/"test.cpp").write_text(r'''
#include "launch_gate.hpp"
#include <cassert>
struct Timer {void begin(void*){calls.push_back(3);} void end(void*){calls.push_back(5);}};
int main() {
  Timer timer;
  { llpython::LaunchGate gate(10000);
    llpython::enqueue(&gate,(void*)3,[]{calls.push_back(2);},timer,[]{calls.push_back(4);});
    assert((calls==std::vector<int>{1,2,3,4,5,6}));
  }
  assert((calls==std::vector<int>{1,2,3,4,5,6,7,8}));
  calls.clear();
  llpython::enqueue< llpython::LaunchGate >(nullptr,(void*)3,[]{calls.push_back(2);},timer,[]{calls.push_back(4);});
  assert((calls==std::vector<int>{2,3,4,5}));
}
''')
    exe = tmp_path/"gate.exe"
    subprocess.run([zig,"c++","-std=c++17","-O1","-UNDEBUG","-I",str(tmp_path),"-I",str(HERE),
                    str(tmp_path/"test.cpp"),"-o",str(exe)],check=True,timeout=180)
    subprocess.run([str(exe)],check=True,timeout=20)


def test_worker_saves_each_case_and_closes_once(tmp_path,monkeypatch):
    run = module("run")
    calls=[]
    class Context:
        npu_stream=11
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Bench:
        def __init__(self,*args): calls.append("init")
        def identity(self): return {"source":"same"}
        def prepare(self,bs): self.bs=bs; calls.append(("prepare",bs))
        def dispatch(self): calls.append("dispatch")
        def clear_flags(self): calls.append("clear")
        def sample(self,cb,device): cb(); calls.append(("sample",device)); return 12.0
        def precision(self): return 0.0
        def close(self): calls.append("close")
    def gather(out,value): out[:]=[value]*len(out)
    def barrier(): calls.append("barrier")
    torch = SimpleNamespace(__version__="test",set_num_threads=lambda n:None,
        npu=SimpleNamespace(set_device=lambda d:None,init=lambda:None,Stream=Context,stream=lambda s:s,synchronize=lambda:None))
    dist = SimpleNamespace(init_process_group=lambda *a,**k:None,all_gather_object=gather,barrier=barrier,
                           destroy_process_group=lambda *a:None)
    torch.distributed=dist
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"torch.distributed",dist)
    monkeypatch.setitem(sys.modules,"torch_npu",SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules,"_ll_py_combine",SimpleNamespace(Bench=Bench,ranks_per_host=4))
    class Gate:
        group=1
        def __init__(self,*args): pass
        def __call__(self): calls.append("gate")
    monkeypatch.setattr(run,"DeviceGate",Gate)
    for key,value in dict(RANK="0",WORLD_SIZE="8",LOCAL_RANK="0",LOCAL_WORLD_SIZE="4").items():
        monkeypatch.setenv(key,value)
    (tmp_path/"bench-build.txt").write_text("source=same\n")
    args=["run.py","--output",str(tmp_path/"out"),"--build",str(tmp_path),"--shmem-ip-port","tcp://a:1",
          "--variant","base","--sync","both","--devices","12,13,14,15","--ranks-per-host","4",
          "--tokens","32,48","--hidden","3584","--topk","8","--experts","384","--capacity","256",
          "--dtype","bf16","--route","random","--seed","0","--warmup","2","--iters","3",
          "--timeout","180","--kernel-timeout","10"]
    monkeypatch.setattr(sys,"argv",args)
    run.main()
    assert calls.count("init")==calls.count("close")==1
    assert calls.count(("sample",True))==10
    assert calls.count(("sample",False))==12  # correctness plus two host-mode batches
    assert len((tmp_path/"out/rank-0.jsonl").read_text().splitlines())==4
    for bs in (32,48):
        for mode in ("host","device"):
            data=json.loads((tmp_path/f"out/bs{bs}-{mode}.json").read_text())
            assert data["rank_max"]["n"]==3 and len(data["per_rank_us"])==8
    assert (tmp_path/"out/rank-0.PASS").is_file()


def test_shell_syntax_and_printed_peer_command(tmp_path):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name=="nt" else shutil.which("bash")
    if not bash: pytest.skip("Bash unavailable")
    for script in HERE.glob("*.sh"):
        subprocess.run([bash,"-n",str(script)],check=True,timeout=10)
    result=subprocess.run([sys.executable,str(HERE/"launch.py"),"--node-rank","0","--dry-run",
                           "--hosts","10.0.0.1,10.0.0.2","--master-port","30123","--sync","both",
                           "--tokens","32,72","--variant","card-all"],text=True,capture_output=True,check=True)
    peer=next(line.split(": ",1)[1] for line in result.stdout.splitlines() if line.startswith("Run on node 1:"))
    peer_args=shlex.split(peer)[2:]
    launch=module("launch")
    args=launch.parser().parse_args(peer_args)
    assert args.node_rank==1 and args.tokens=="32,72" and args.sync=="both" and args.master_port==30123
