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
    assert node==0 and len(hosts)==2 and build.name=="base" and build.parent.name=="deepep_ll_bench"
    assert str(build).startswith(sys.prefix)
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
    for variant in ("base", "input6", "input-interleave", "input-l2"):
        for hosts in ("a,b", "a,b,c,d"):
            args = launch.parser().parse_args(["same-run", "--hosts", hosts, "--node-rank", "1",
                                              "--ranks-per-host", "16", "--variant", variant, "--sync", "hccl"])
            _, _, _, _, build, cmd = launch.plan(args)
            assert build.name == variant
            assert cmd[cmd.index("--devices")+1] == devices
            assert cmd[cmd.index("--sync")+1] == "hccl"


def test_clean_build_suite_forwards_environment_and_stops(tmp_path):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists(): pytest.skip("Bash unavailable")
    scripts = tmp_path/"repo with spaces/tests/ll_python_bench"
    scripts.mkdir(parents=True)
    cpp = scripts.parent/"ll_cpp_bench"
    cpp.mkdir()
    shutil.copyfile(HERE/"build_all.sh",scripts/"build_all.sh")
    shutil.copyfile(HERE.parent/"ll_cpp_bench/experiment_defaults.sh",cpp/"experiment_defaults.sh")
    (scripts/"build.sh").write_text('''#!/usr/bin/env bash
set -eu
printf 'CALL %s %s %s %s %s %s\\n' "$@"
if [[ "$2" == "${FAIL_VARIANT:-}" ]]; then exit 17; fi
''',newline="\n")
    cmd = [bash,str(scripts/"build_all.sh"),"--ranks-per-host","16","--conda-env","py311_zyq"]
    result = subprocess.run(cmd,capture_output=True,text=True,check=True)
    calls = result.stdout.splitlines()
    assert calls == [f"CALL --variant {v} --ranks-per-host 16 --conda-env py311_zyq"
                     for v in ("base","input6","input-interleave","input-l2")]
    failed = subprocess.run(cmd,env=dict(os.environ,FAIL_VARIANT="input6"),capture_output=True,text=True)
    assert failed.returncode == 17 and "input-interleave" not in failed.stdout
    invalid = subprocess.run(cmd+["--variants","base,typo"],capture_output=True,text=True)
    assert invalid.returncode == 2 and not invalid.stdout


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
    assert result["worst"][0]==dict(iteration=2,rank=0,us=100)
    assert result["first_means"]["3"]==pytest.approx(200/3)
    assert result["rank_p50_range"]==[12,30]
    data = [[1.0]*50+[2.0]*50,[3.0]*100]
    assert run.summarize(data)["rank_max_tail50"]["n"]==50
    for bad in ([[1],[1,2]],[[1,float("nan")]],[[0]],[]):
        with pytest.raises(ValueError): run.summarize(bad)


def test_worker_environment_disables_blocking_wait(tmp_path,monkeypatch):
    launch=module("launch")
    masks=("ASCEND_RT_VISIBLE_DEVICES","ASCEND_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")
    for key in ("TORCH_HCCL_BLOCKING_WAIT","HCCL_BLOCKING_WAIT",*masks):
        monkeypatch.setenv(key,"1")
    env=launch.worker_environment(launch.parser().parse_args([]),tmp_path)
    assert env["TORCH_HCCL_BLOCKING_WAIT"]==env["HCCL_BLOCKING_WAIT"]=="0"
    assert all(key not in env for key in masks)
    assert env["PYTHONFAULTHANDLER"]=="1"


@pytest.mark.parametrize("nodes",[2,4])
def test_gloo_auto_selects_each_hosts_entry(nodes,monkeypatch):
    launch=module("launch")
    hosts=["90.90.93."+v for v in ("87","86","83","81")][:nodes]
    args=launch.parser().parse_args([])
    for node,host in enumerate(hosts):
        # The machine name resolves to loopback; selection must use --hosts.
        monkeypatch.setattr(launch.socket,"getaddrinfo",lambda name,*a:[(0,0,0,"",(name,0))])
        interfaces=[dict(ifname="lo",flags=["UP","LOOPBACK"],addr_info=[dict(local="127.0.0.1")]),
                    dict(ifname="wrong",flags=["UP"],addr_info=[dict(local="192.168.1.1")]),
                    dict(ifname=f"eth{node}",flags=["UP"],addr_info=[dict(local=host)])]
        def ip_command(cmd,**kwargs):
            assert cmd==["ip","-j","-4","address","show"] and kwargs["timeout"]==5
            return json.dumps(interfaces)
        monkeypatch.setattr(launch.subprocess,"check_output",ip_command)
        env={"HCCL_SOCKET_IFNAME":"hccl-control"}
        line=launch.configure_gloo_network(args,node,hosts,env)
        assert env==dict(GLOO_SOCKET_IFNAME=f"eth{node}",HCCL_SOCKET_IFNAME="hccl-control")
        assert f"host={host}" in line and f"source=host:{host}" in line


@pytest.mark.parametrize("mode",["loopback","missing","ambiguous","down","no-ip"])
def test_gloo_auto_selection_fails_before_launch(mode,monkeypatch):
    launch=module("launch")
    args=launch.parser().parse_args([])
    host="90.90.93.83"
    monkeypatch.setattr(launch.socket,"getaddrinfo",lambda *a:[(0,0,0,"",("127.0.0.1" if mode=="loopback" else host,0))])
    def ip_command(*a,**kw):
        if mode=="no-ip": raise FileNotFoundError("ip unavailable")
        return [dict(ifname=name,flags=[] if mode=="down" else ["UP"],addr_info=[dict(local=host)])
                for name in ([] if mode=="missing" else ["eth0","eth1"] if mode=="ambiguous" else ["eth0"])]
    monkeypatch.setattr(launch.subprocess,"check_output",lambda *a,**kw:json.dumps(ip_command()))
    env={}
    with pytest.raises(ValueError,match="--ifname"):
        launch.configure_gloo_network(args,1,["90.90.93.87",host],env)
    assert "GLOO_SOCKET_IFNAME" not in env


@pytest.mark.parametrize("explicit,inherited,ok",[(None,"eth1",True),("eth0","eth1",True),
                                                (None,"eth0,eth1",True),(None,"lo",False),
                                                ("absent",None,False),(None,"eth0,",False)])
def test_gloo_explicit_interface_precedence(tmp_path,monkeypatch,explicit,inherited,ok):
    launch=module("launch")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME",raising=False)
    monkeypatch.delenv("HCCL_SOCKET_IFNAME",raising=False)
    if inherited: monkeypatch.setenv("GLOO_SOCKET_IFNAME",inherited)
    monkeypatch.setattr(launch.socket,"if_nameindex",lambda:[(1,"lo"),(2,"eth0"),(3,"eth1")])
    def forbidden(*a,**kw): raise AssertionError("Explicit interface does not need ip/DNS discovery")
    monkeypatch.setattr(launch.subprocess,"check_output",forbidden)
    args=launch.parser().parse_args(["--ifname",explicit] if explicit else [])
    env=launch.worker_environment(args,tmp_path)
    if not ok:
        with pytest.raises(ValueError,match="GLOO_SOCKET_IFNAME"):
            launch.configure_gloo_network(args,0,["a","b"],env)
        return
    launch.configure_gloo_network(args,0,["a","b"],env)
    assert env["GLOO_SOCKET_IFNAME"]==(explicit or inherited)
    assert env.get("HCCL_SOCKET_IFNAME")==explicit


def test_gloo_single_host_keeps_default(monkeypatch):
    launch=module("launch")
    def forbidden(*a,**kw): raise AssertionError("Single-host default does not require network discovery")
    monkeypatch.setattr(launch.subprocess,"check_output",forbidden)
    env={}
    line=launch.configure_gloo_network(launch.parser().parse_args([]),0,["localhost"],env)
    assert env=={} and "GLOO_SOCKET_IFNAME=auto" in line


def test_launch_passes_and_logs_selected_network(tmp_path,monkeypatch,capsys):
    launch=module("launch")
    monkeypatch.setattr(launch,"ROOT",tmp_path)
    monkeypatch.setattr(sys,"argv",["launch.py","network-run","--node-rank","2","--hosts","87,86,83,81", "--check-hccl"])
    monkeypatch.delenv("GLOO_SOCKET_IFNAME",raising=False)
    monkeypatch.setitem(sys.modules,"resource",SimpleNamespace(RLIMIT_NOFILE=0,getrlimit=lambda _: (65536,65536)))
    monkeypatch.setattr(launch.socket,"getaddrinfo",lambda *a:[(0,0,0,"",("90.90.93.83",0))])
    monkeypatch.setattr(launch.subprocess,"check_output",lambda *a,**kw:json.dumps([
        dict(ifname="eth2",flags=["UP"],addr_info=[dict(local="90.90.93.83")])]))
    def start(cmd,**kwargs):
        assert kwargs["env"]["GLOO_SOCKET_IFNAME"]=="eth2"
        assert "--node_rank=2" in cmd and "--nnodes=4" in cmd
        return SimpleNamespace(stdout=["probe completed\n"],wait=lambda:0)
    monkeypatch.setattr(launch.subprocess,"Popen",start)
    assert launch.main()==0
    line=(tmp_path/"results/ll-python/network-run/node-2/console.log").read_text().splitlines()[0]
    assert "GLOO_SOCKET_IFNAME=eth2" in line and line in capsys.readouterr().out


def test_hccl_probe_never_loads_native_or_shmem(tmp_path,monkeypatch,capsys):
    run=module("run")
    calls=[]
    torch=SimpleNamespace(__version__="test",set_num_threads=lambda n:None,
        npu=SimpleNamespace(set_device=lambda d:calls.append(("device",d)),init=lambda:None,
                            current_device=lambda:12,synchronize=lambda:calls.append("sync")))
    dist=SimpleNamespace(init_process_group=lambda *a,**kw:None,
        all_gather_object=lambda out,v:out.__setitem__(slice(None),[v]*len(out)),
        barrier=lambda:None,destroy_process_group=lambda *a:None)
    torch.distributed=dist
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"torch.distributed",dist)
    monkeypatch.setitem(sys.modules,"torch_npu",SimpleNamespace(__version__="test"))
    class Gate:
        group=1
        def __init__(self,*args,**kwargs): calls.append("hccl-init")
        def __call__(self): calls.append("reduce")
    monkeypatch.setattr(run,"DeviceGate",Gate)
    def forbidden(*args): raise AssertionError("Probe must not load any native modules")
    monkeypatch.setattr(run,"load_native",forbidden)
    for key,value in dict(RANK="0",WORLD_SIZE="8",LOCAL_RANK="0",LOCAL_WORLD_SIZE="4").items():
        monkeypatch.setenv(key,value)
    launch=module("launch")
    args=launch.parser().parse_args(["--node-rank","0","--check-hccl"])
    cmd=launch.plan(args)[-1]
    worker=cmd[cmd.index(str(HERE/"run.py")):] if str(HERE/"run.py") in cmd else cmd[next(i for i,v in enumerate(cmd) if v.endswith("run.py")):]
    worker[worker.index("--output")+1]=str(tmp_path)
    monkeypatch.setattr(sys,"argv",worker)
    run.main()
    assert calls[0]==("device",12) and calls.count("hccl-init")==1 and calls.count("reduce")==3
    assert (tmp_path/"rank-0.HCCL_PASS").exists() and not (tmp_path/"rank-0.PASS").exists()
    assert "no Combine/SHMEM loaded" in capsys.readouterr().out


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
    stages=[]
    gate = run.DeviceGate(torch,dist,180,progress=stages.append)
    assert calls==["reduce","wait","drain","sync"]
    assert stages==["HCCL group creation","HCCL token allocation","HCCL first all_reduce",
                    "HCCL initial stream synchronization","HCCL gate ready"]
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


def test_configuration_diff_groups_fields_and_ranks():
    run = module("run")
    records = [dict(source_id="old",iters=100,extra=None)]*4 + [dict(source_id="new",iters=100)]*4
    assert run.configuration_diff(records)==[
        'extra: ranks=0,1,2,3 value=null; ranks=4,5,6,7 value=<missing>',
        'source_id: ranks=0,1,2,3 value="old"; ranks=4,5,6,7 value="new"']
    assert run.configuration_diff([dict(iters=100),dict(iters=100)])==[]


def test_python_patch_difference_does_not_hide_other_mismatches():
    run = module("run")
    records = [dict(python="3.11.17",source_id="same",torch="2.11.0",tokens="32,48")]*4
    records += [dict(records[0],python="3.11.18")]*4
    assert run.configuration_diff(records,allow_python_patch=True)==[]
    assert "3.11.17" in run.configuration_diff(records)[0]
    for field,value in (("python","3.12.0"),("python","4.11.17"),
                        ("source_id","other"),("torch","2.10.0"),("tokens","32")):
        changed=records[:4]+[dict(records[4],**{field:value})]*4
        assert run.configuration_diff(changed,allow_python_patch=True)[0].startswith(field+":")


@pytest.mark.parametrize("failure",[None,"python_patch","finalize","configuration","native"])
@pytest.mark.parametrize("variants", [("base",), ("base","input6","input-interleave","input-l2")])
@pytest.mark.parametrize("sync", ["both", "hccl"])
def test_worker_saves_each_case_and_closes_once(tmp_path,monkeypatch,capsys,failure,variants,sync):
    run = module("run")
    calls=[]
    class Context:
        npu_stream=11
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Bench:
        def __init__(self,*args):
            self.owner = len(args)>1
            if self.owner:
                assert args[3][args[3].index("--devices")+1]=="12,13,14,15"
                calls.append("init")
            else: calls.append("borrow")
        def share(self): return "capsule"
        def reset_workspace(self): calls.append("reset")
        def release_case(self): calls.append("release-case")
        def identity(self): return dict(source_id="same",active_aiv=56,recv_cores=28,rdma_qps=56,return_options=0)
        def prepare(self,bs): self.bs=bs; calls.append(("prepare",bs))
        def dispatch(self): calls.append("dispatch")
        def clear_flags(self): calls.append("clear")
        def check_entry_sync(self): calls.append("entry-preflight")
        def entry_sync(self): calls.append("device-barrier")
        def sample(self,cb,device): cb(); calls.append(("sample",device)); return 12.0
        def precision(self): return 0.0
        def close(self):
            calls.append("close" if self.owner else "release")
            if self.owner and failure=="finalize": raise RuntimeError("injected finalize failure")
    def gather(out,value):
        out[:]=[value]*len(out)
        stage = "configuration" if "tokens" in value else "native"
        if failure==stage and "source_id" in value:
            out[4:] = [dict(value,source_id="other")]*4
        elif failure=="python_patch" and stage=="configuration":
            out[4:] = [dict(value,python=".".join(value["python"].split(".")[:2])+".999")]*4
    def barrier(): calls.append("barrier")
    def set_device(device):
        assert device==12  # local rank 0 must use selected card 12, never card 0
        calls.append("set_device")
    torch = SimpleNamespace(__version__="test",set_num_threads=lambda n:None,
        npu=SimpleNamespace(set_device=set_device,current_device=lambda:12,init=lambda:calls.append("npu_init"),
                            Stream=Context,stream=lambda s:s,synchronize=lambda:None))
    dist = SimpleNamespace(init_process_group=lambda *a,**k:None,all_gather_object=gather,barrier=barrier,
                           destroy_process_group=lambda *a:None)
    torch.distributed=dist
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"torch.distributed",dist)
    monkeypatch.setitem(sys.modules,"torch_npu",SimpleNamespace(__version__="test"))
    def load_native(*args):
        calls.append("load-native")
        return SimpleNamespace(Bench=Bench,ranks_per_host=4)
    monkeypatch.setattr(run,"load_native",load_native)
    class Gate:
        group=1
        def __init__(self,*args,**kwargs):
            assert sync=="hccl"
            assert "init" not in calls and "load-native" not in calls
            assert "set_device" in calls
            calls.append("hccl-init")
        def __call__(self): calls.append("gate")
    monkeypatch.setattr(run,"DeviceGate",Gate)
    for key,value in dict(RANK="0",WORLD_SIZE="8",LOCAL_RANK="0",LOCAL_WORLD_SIZE="4").items():
        monkeypatch.setenv(key,value)
    for variant in variants:
        (tmp_path/variant).mkdir()
        (tmp_path/variant/"bench-build.txt").write_text("source=same\n")
    args=["run.py","--output",str(tmp_path/"out"),"--module-dir",str(tmp_path/"base"),"--variants",",".join(variants),"--shmem-ip-port","tcp://a:1",
          "--variant","base","--sync",sync,"--devices","12,13,14,15","--ranks-per-host","4",
          "--tokens","32,48","--hidden","3584","--topk","8","--experts","384","--capacity","256",
          "--dtype","bf16","--route","random","--seed","0","--warmup","2","--iters","3",
          "--timeout","180","--kernel-timeout","10"]
    monkeypatch.setattr(sys,"argv",args)
    if failure in ("configuration","native"):
        with pytest.raises(run.ConfigurationMismatch,match=failure): run.main()
        assert calls.count("init")==int(failure=="native")
        assert calls.count("set_device")==calls.count("npu_init")==int(failure=="native")
        assert not any(isinstance(call,tuple) and call[0]=="sample" for call in calls)
        snapshot=json.loads((tmp_path/("out" if failure=="configuration" else "out/base")/f"rank-0-{failure}.json").read_text())
        assert snapshot[0]["source_id"]==({v:"same" for v in variants} if failure=="configuration" else "same")
        assert snapshot[4]["source_id"]=="other"
        console=capsys.readouterr().out
        assert f"LL_PY_MISMATCH_BEGIN stage={failure}" in console
        assert 'source_id: ranks=0,1,2,3 value=' in console and 'ranks=4,5,6,7 value="other"' in console
        assert not (tmp_path/"out/rank-0.PASS").exists()
        return
    fail_close = failure=="finalize"
    if fail_close:
        with pytest.raises(RuntimeError,match="finalize failure"): run.main()
    else:
        run.main()
    assert calls.count("init")==calls.count("close")==1
    assert calls.count("borrow")==calls.count("release")==calls.count("reset")==len(variants)-1
    assert calls.count("release-case")==1
    assert calls.count(("sample",True))==10*len(variants)
    assert calls.count(("sample",False))==(12 if sync=="both" else 2)*len(variants)  # correctness plus two host-mode batches
    assert calls.count("hccl-init")==int(sync=="hccl")
    assert calls.count("entry-preflight")==len(variants)*int(sync=="both")
    assert calls.count("device-barrier")==10*len(variants)*int(sync=="both")
    assert len((tmp_path/"out/rank-0.jsonl").read_text().splitlines())==(4 if sync=="both" else 2)*len(variants)
    for bs in (32,48):
        for mode in (("host","device") if sync=="both" else ("hccl",)):
            data=json.loads((tmp_path/f"out/base/bs{bs}-{mode}.json").read_text())
            assert data["rank_max"]["n"]==3 and len(data["per_rank_us"])==8
            assert data["sync_impl"]==run.sync_implementation(mode)
    assert (tmp_path/"out/rank-0.PASS").is_file()==(not fail_close)
    report=(tmp_path/"out/report.txt").read_text()
    assert f"status={'RUNNING' if fail_close else 'PASS'}" in report
    assert "Event_us=per-iteration rank-MAX" in report and "aiv=56 recv=28 qps=56" in report
    assert "device=SHMEM-mixed-71a2154" in report
    if sync=="both": assert "AB BS=32 variant=base device_p50_reduction=0.00%" in report
    assert f"D BS=48 variant=base sync={'device' if sync=='both' else 'hccl'}" in report and "i1/r0:12.00" in report
    console=capsys.readouterr().out
    assert "samples 0/3" not in console  # sampling progress is only printed on the heartbeat
    assert ("LL_PY_REPORT_BEGIN" in console)==(not fail_close)
    if failure=="python_patch":
        assert "INFO allowed Python patch difference:" in console
        snapshot=json.loads((tmp_path/"out/rank-0-configuration.json").read_text())
        assert snapshot[0]["python"]==sys.version.split()[0]
        assert snapshot[4]["python"].endswith(".999")


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


@pytest.mark.parametrize("wrong_python",[False,True])
def test_conda_activation_selects_and_verifies_python(tmp_path,wrong_python):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name=="nt" else shutil.which("bash")
    if not bash: pytest.skip("Bash unavailable")
    target=tmp_path/"envs/py311_zyq"
    (target/"bin").mkdir(parents=True)
    sdk=tmp_path/"tests/ll_cpp_bench"
    sdk.mkdir(parents=True)
    (sdk/"experiment_defaults.sh").write_text('ll_bench_environment() { :; }\n')
    conda=tmp_path/"conda"
    conda.write_text('''#!/usr/bin/env bash
cat <<'HOOK'
conda() {
    [[ $1 == activate && $2 == py311_zyq ]] || return 3
    export CONDA_PREFIX="$TEST_ENV" CONDA_DEFAULT_ENV=py311_zyq
    echo activated >> "$TEST_TRACE"
}
HOOK
''')
    # Execute the real environment validation code with a simulated Conda prefix.
    python=target/"bin/python"
    python.write_text('''#!/usr/bin/env bash
echo selected >> "$TEST_TRACE"
exec "$TEST_PYTHON" -c 'import sys,os; sys.prefix=os.environ["TEST_PREFIX"]; sys.argv=["-",sys.argv[1]]; exec(sys.stdin.read())' "$2"
''')
    for script in (conda,python): script.chmod(0o755)
    trace=tmp_path/"trace"
    env=dict(os.environ,CONDA_DEFAULT_ENV="base",CONDA_PREFIX=str(tmp_path/"base"),CONDA_EXE=conda.as_posix(),
             TEST_ENV=target.as_posix(),TEST_PREFIX=str(tmp_path/"wrong" if wrong_python else target),
             TEST_PYTHON=Path(sys.executable).as_posix(),TEST_TRACE=trace.as_posix())
    env.pop("PYTHONHOME",None)
    cmd=f'''set -euo pipefail
root={shlex.quote(tmp_path.as_posix())}
cann=unused mpi=unused shmem=unused
source {shlex.quote((HERE/'environment.sh').as_posix())}
ll_python_environment py311_zyq
'''
    result=subprocess.run([bash,"-c",cmd],env=env,text=True,capture_output=True,timeout=20)
    assert trace.read_text().splitlines()==["activated","selected"]
    if wrong_python:
        assert result.returncode!=0 and "Wrong Python environment" in result.stderr
    else:
        assert result.returncode==0,result.stdout+result.stderr
        assert "ENV=py311_zyq" in result.stdout
