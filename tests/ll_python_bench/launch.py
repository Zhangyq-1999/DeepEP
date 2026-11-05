"""One torchrun agent per host; defaults match the current cross-host EP8 case."""
import argparse
import datetime
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", nargs="?")
    p.add_argument("--hosts", default="90.90.93.84,90.90.93.184")
    p.add_argument("--node-rank", type=int)
    p.add_argument("--master-port", type=int, default=30011)
    p.add_argument("--shmem-ip-port", help="SHMEM TCP endpoint; default tcp://HOST0:30012")
    p.add_argument("--devices", help="Default: 12..15 for four ranks, 0..15 for sixteen ranks")
    p.add_argument("--ranks-per-host", type=int, choices=(4, 16), default=4)
    p.add_argument("--variant", choices=("base", "ready-snapshot", "wqe-prep", "wqe-prefetch", "mte-pack",
                                         "card-all", "card-row", "server-all", "server-row"), default="base")
    p.add_argument("--sync", choices=("device", "host", "both", "hccl"), default="device",
                   help="device=SHMEM barrier; both=Gloo vs SHMEM; hccl=explicit legacy HCCL comparison")
    p.add_argument("--tokens", default="32,48,64,72,256")
    for name, default in (("hidden",3584),("topk",8),("experts",384),("capacity",256),
                          ("seed",0),("warmup",10),("iters",100),("timeout",180),("kernel-timeout",10)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--route", choices=("random", "hotspot"), default="random")
    p.add_argument("--gate-delay-ms", type=float, default=0, help="Untimed rank-0 delay to check gate behavior")
    p.add_argument("--ifname", help="Optional GLOO_SOCKET_IFNAME and HCCL_SOCKET_IFNAME")
    p.add_argument("--dry-run", action="store_true")
    return p


def local_node(hosts):
    names = {socket.gethostname(), socket.getfqdn()}
    try:
        names.update(subprocess.check_output(["hostname", "-I"], text=True).split())
    except (OSError, subprocess.SubprocessError):
        pass
    matches = [i for i, host in enumerate(hosts) if host in names]
    if len(matches) != 1:
        raise ValueError("Cannot detect node; pass --node-rank 0/1 (or 0..3 for four hosts)")
    return matches[0]


def plan(a):
    hosts = a.hosts.split(",")
    if a.devices is None:
        a.devices = "12,13,14,15" if a.ranks_per_host == 4 else ",".join(map(str,range(16)))
    devices = [int(v) for v in a.devices.split(",")]
    if len(hosts) not in (1,2,4) or len(set(hosts)) != len(hosts):
        raise ValueError("Use one, two or four distinct hosts")
    if len(devices) != a.ranks_per_host or len(set(devices)) != len(devices) or min(devices)<0:
        raise ValueError("List one distinct device per local rank")
    node = local_node(hosts) if a.node_rank is None else a.node_rank
    if not 0 <= node < len(hosts) or not 1024 <= a.master_port <= 65534:
        raise ValueError("Invalid node rank or master port")
    if a.iters<1 or a.warmup<0 or a.timeout<1 or a.kernel_timeout<1 or not 0<=a.gate_delay_ms<=60000:
        raise ValueError("Invalid sample count or timeout/delay")
    if a.gate_delay_ms and a.sync == "host":
        raise ValueError("Gate delay validation requires device synchronization")
    run_id = a.run_id
    if not run_id:
        if node: raise ValueError("Copy RUN_ID from node 0")
        run_id = f"ep{len(hosts)*len(devices)}-py-{datetime.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"
    if run_id in (".","..") or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in run_id):
        raise ValueError("Invalid RUN_ID")
    endpoint = a.shmem_ip_port or f"tcp://{hosts[0]}:{a.master_port+1}"
    out = ROOT / "results/ll-python" / run_id
    build = Path(sysconfig.get_path("platlib")) / "deepep_ll_bench" / a.variant
    cmd = [sys.executable,"-u","-m","torch.distributed.run",f"--nnodes={len(hosts)}",
           f"--nproc_per_node={len(devices)}",f"--node_rank={node}",f"--master_addr={hosts[0]}",
           f"--master_port={a.master_port}","--max_restarts=0",str(Path(__file__).with_name("run.py")),
           "--output",str(out),"--module-dir",str(build),"--shmem-ip-port",endpoint,
           "--variant",a.variant,"--sync",a.sync,"--devices",a.devices,"--ranks-per-host",str(a.ranks_per_host)]
    for key in ("tokens","hidden","topk","experts","capacity","dtype","route","seed","warmup","iters",
                "timeout","kernel_timeout","gate_delay_ms"):
        cmd += ["--"+key.replace("_","-"),str(getattr(a,key))]
    return node, hosts, run_id, out, build, cmd


def worker_environment(a, node_dir):
    env = dict(os.environ, DEEPEP_LL_BACKEND="rdma",
               PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", PYTHONFAULTHANDLER="1", OMP_NUM_THREADS="1",
               HCCL_CONNECT_TIMEOUT=str(a.timeout), HCCL_EXEC_TIMEOUT=str(a.timeout),
               # Work.wait must enqueue a stream dependency, not block the Host
               # while the local Notify is still holding that stream.
               TORCH_HCCL_BLOCKING_WAIT="0", HCCL_BLOCKING_WAIT="0")
    if a.ifname: env.update(GLOO_SOCKET_IFNAME=a.ifname,HCCL_SOCKET_IFNAME=a.ifname)
    # HCCL topology discovery can query devices outside the selected subset.
    # Select each worker's device explicitly, without hiding/remapping devices.
    for key in ("ASCEND_RT_VISIBLE_DEVICES","ASCEND_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES"):
        env.pop(key,None)
    env["ASCEND_PROCESS_LOG_PATH"] = str(node_dir/"ascend")
    return env


def main():
    a = parser().parse_args()
    node, hosts, run_id, out, build, cmd = plan(a)
    print(f"RUN_ID={run_id}\nnode={node} EP={len(hosts)*a.ranks_per_host} variant={a.variant} sync={a.sync}",flush=True)
    print(f"devices={a.devices} BS={a.tokens} warmup={a.warmup} iters={a.iters}\nResults: {out}",flush=True)
    if node == 0:
        custom = []
        defaults = vars(parser().parse_args([]))
        defaults["devices"] = "12,13,14,15" if a.ranks_per_host == 4 else ",".join(map(str,range(16)))
        for key in ("hosts","master_port","shmem_ip_port","devices","ranks_per_host","variant","sync","tokens",
                    "hidden","topk","experts","capacity","dtype","route","seed","warmup","iters","timeout",
                    "kernel_timeout","gate_delay_ms","ifname"):
            value = getattr(a,key)
            if value != defaults[key]: custom += ["--"+key.replace("_","-"),str(value)]
        for peer in range(1,len(hosts)):
            tail = custom + (["--node-rank",str(peer)] if a.node_rank is not None else [])
            if os.environ.get("LL_PY_CONDA_ENV","py311_zyq") != "py311_zyq":
                tail += ["--conda-env",os.environ["LL_PY_CONDA_ENV"]]
            print(f"Run on node {peer}: "+shlex.join(["bash","tests/ll_python_bench/run_all.sh",run_id,*tail]),flush=True)
    if a.dry_run:
        print(shlex.join(cmd)); return 0
    if not list(build.glob("_ll_py_combine*.so")):
        raise RuntimeError(f"Install into {sys.prefix} first: bash tests/ll_python_bench/build.sh --variant {a.variant}")
    if f"ranks_per_host={a.ranks_per_host}" not in (build/"bench-build.txt").read_text().splitlines():
        raise RuntimeError("Rebuild with matching --ranks-per-host")
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 65536: resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard) if hard>=0 else 65536,hard))
    node_dir = out / f"node-{node}"
    node_dir.mkdir(parents=True,exist_ok=False)
    env = worker_environment(a,node_dir)
    # Console is also saved locally. torchrun fails the local worker group on first error.
    with (node_dir/"console.log").open("w") as log:
        process = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end="",flush=True); log.write(line); log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate(); process.wait(timeout=30); return 130


if __name__ == "__main__":
    try: sys.exit(main())
    except (ValueError,RuntimeError) as exc: raise SystemExit(f"ERROR: {exc}")
