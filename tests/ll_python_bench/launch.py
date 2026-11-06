"""One torchrun agent per host; defaults match the current cross-host EP8 case."""
import argparse
import datetime
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = "base,input6-l2-stage,input6-l2-all,input6-l2-pipe"
CLEAN_VARIANTS = ("base", "ready-snapshot", "wqe-prep", "wqe-prefetch", "input6", "input-ready", "input-pair", "input-pipe", "input-stream", "input-front", "input-interleave", "input-l2", "input6-l2", "input6-r32", "input6-remote", "input6-fused", "input6-rpipe", "input6-rfull", "mte-pack",
                  "card-all", "card-row", "server-all", "server-row", "input6-l2-remote", "input6-l2-stage", "input6-l2-tx", "input6-l2-rx", "input6-l2-all", "input6-l2-pipe")


def phase_names(a):
    clean = a.variants.split(",")
    return clean + (["ref-profile" if name=="base" else name+"-profile" for name in clean]
                    if a.profile and not a.check_hccl else [])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", nargs="?")
    p.add_argument("--hosts", default="90.90.93.84,90.90.93.184")
    p.add_argument("--node-rank", type=int)
    p.add_argument("--master-port", type=int, default=30011)
    p.add_argument("--shmem-ip-port", help="SHMEM TCP endpoint; default tcp://HOST0:30012")
    p.add_argument("--devices", help="Default: 12..15 for four ranks, 0..15 for sixteen ranks")
    p.add_argument("--ranks-per-host", type=int, choices=(4, 16), default=4)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--variant", choices=CLEAN_VARIANTS, help="Run only this variant")
    selection.add_argument("--variants", help="Comma-separated suite; default: "+DEFAULT_VARIANTS)
    p.add_argument("--check-hccl", action="store_true", help="Test Torch/HCCL alone without native/SHMEM initialization")
    p.add_argument("--sync", choices=("device", "host", "both", "hccl"), default="device",
                   help="device=SHMEM barrier; both=Gloo vs SHMEM; hccl=explicit legacy HCCL comparison")
    p.add_argument("--tokens", default="32,48,64,72,256")
    p.add_argument("--profile", type=int, choices=(0,1,2), default=2, help="Append separate diagnostic phases; 0=clean only")
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument("--profile-frequency-hz", type=int, default=1000000000, help="Device clock for Perfetto; 0=CSV only")
    for name, default in (("hidden",3584),("topk",8),("experts",384),("capacity",256),
                          ("seed",0),("warmup",10),("iters",100),("timeout",180),("kernel-timeout",10)):
        p.add_argument("--"+name, type=int, default=default)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--route", choices=("random", "hotspot"), default="random")
    p.add_argument("--gate-delay-ms", type=float, default=0, help="Untimed rank-0 delay to check gate behavior")
    p.add_argument("--ifname", help="Override Gloo/HCCL interfaces; otherwise Gloo uses this node's --hosts IP")
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
    a.variants = a.variant or a.variants or DEFAULT_VARIANTS
    names = a.variants.split(",")
    if len(names)!=len(set(names)) or any(v not in CLEAN_VARIANTS for v in names):
        raise ValueError("Use distinct clean variants: "+",".join(CLEAN_VARIANTS))
    if a.check_hccl: a.sync = "hccl"
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
    if not 1<=a.profile_iters<=10000 or not 0<=a.profile_frequency_hz<=0xffffffff:
        raise ValueError("Invalid profile samples or frequency")
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
    build = Path(sysconfig.get_path("platlib")) / "deepep_ll_bench" / names[0]
    cmd = [sys.executable,"-u","-m","torch.distributed.run",f"--nnodes={len(hosts)}",
           f"--nproc_per_node={len(devices)}",f"--node_rank={node}",f"--master_addr={hosts[0]}",
           f"--master_port={a.master_port}","--max_restarts=0",str(Path(__file__).with_name("run.py")),
           "--output",str(out),"--module-dir",str(build),"--shmem-ip-port",endpoint,
           "--variant",names[0],"--variants",",".join(phase_names(a)),"--sync",a.sync,"--devices",a.devices,
           "--ranks-per-host",str(a.ranks_per_host)]
    if a.check_hccl: cmd.append("--check-hccl")
    for key in ("tokens","hidden","topk","experts","capacity","dtype","route","seed","warmup","iters",
                "timeout","kernel_timeout","gate_delay_ms","profile","profile_iters","profile_frequency_hz"):
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


def configure_gloo_network(a, node, hosts, env):
    """Select a routable local interface before torchrun imports Torch.

    MASTER_ADDR only selects the rendezvous store; it does not select Gloo's
    full-mesh transport address. Hostname lookup can otherwise select 127.0.0.1.
    Keep explicit interface settings and HCCL's independent default intact.
    """
    chosen = (a.ifname or env.get("GLOO_SOCKET_IFNAME", "")).strip()
    source = "--ifname" if a.ifname else "environment"
    if chosen:
        names = chosen.split(",")
        available = {name for _, name in socket.if_nameindex()}
        if len(set(names)) != len(names) or any(not n or n not in available for n in names):
            raise ValueError(f"Invalid GLOO_SOCKET_IFNAME={chosen}; local interfaces: {','.join(sorted(available))}")
        if len(hosts) > 1 and "lo" in names:
            raise ValueError("Multi-host Gloo cannot use lo; unset GLOO_SOCKET_IFNAME for automatic selection or pass --ifname")
    elif len(hosts) > 1:
        source = f"host:{hosts[node]}"
        try:
            candidates = {entry[4][0] for entry in socket.getaddrinfo(hosts[node], None, socket.AF_INET, socket.SOCK_STREAM)}
            candidates = {ip for ip in candidates if not ipaddress.ip_address(ip).is_loopback
                          and not ipaddress.ip_address(ip).is_unspecified}
            if not candidates:
                raise ValueError(f"--hosts entry {hosts[node]} has no non-loopback IPv4 address")
            interfaces = json.loads(subprocess.check_output(["ip", "-j", "-4", "address", "show"], text=True, timeout=5))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ValueError(f"Cannot select Gloo interface for {hosts[node]}: {exc}; set --ifname to this node's network interface") from exc
        matches = {item["ifname"] for item in interfaces
                   if "UP" in item.get("flags", []) and "LOOPBACK" not in item.get("flags", [])
                   and any(addr.get("local") in candidates for addr in item.get("addr_info", []))}
        if len(matches) != 1:
            raise ValueError(f"Expected one UP local interface for --hosts[{node}]={hosts[node]}, found {sorted(matches)}; "
                             "check --node-rank/--hosts or set --ifname")
        chosen = matches.pop()
    else:
        source = "single-host"
    if chosen:
        env["GLOO_SOCKET_IFNAME"] = chosen
    return (f"Network: node={node} host={hosts[node]} GLOO_SOCKET_IFNAME={chosen or 'auto'} "
            f"source={source} HCCL_SOCKET_IFNAME={env.get('HCCL_SOCKET_IFNAME') or 'auto'}")


def main():
    a = parser().parse_args()
    node, hosts, run_id, out, build, cmd = plan(a)
    print(f"RUN_ID={run_id}\nnode={node} EP={len(hosts)*a.ranks_per_host} variants={a.variants} sync={a.sync}"
          f" check_hccl={int(a.check_hccl)}",flush=True)
    print(f"devices={a.devices} BS={a.tokens} warmup={a.warmup} clean_iters={a.iters} "
          f"profile={a.profile} profile_iters={a.profile_iters}\nResults: {out}",flush=True)
    if node == 0:
        custom = []
        defaults = vars(parser().parse_args([]))
        defaults["variants"] = DEFAULT_VARIANTS
        defaults["devices"] = "12,13,14,15" if a.ranks_per_host == 4 else ",".join(map(str,range(16)))
        for key in ("hosts","master_port","shmem_ip_port","devices","ranks_per_host","variants","sync","tokens",
                    "hidden","topk","experts","capacity","dtype","route","seed","warmup","iters","timeout",
                    "kernel_timeout","gate_delay_ms","ifname","profile","profile_iters","profile_frequency_hz"):
            value = getattr(a,key)
            if value != defaults[key]: custom += ["--"+key.replace("_","-"),str(value)]
        if a.check_hccl: custom += ["--check-hccl"]
        for peer in range(1,len(hosts)):
            tail = custom + (["--node-rank",str(peer)] if a.node_rank is not None else [])
            if os.environ.get("LL_PY_CONDA_ENV","py311_zyq") != "py311_zyq":
                tail += ["--conda-env",os.environ["LL_PY_CONDA_ENV"]]
            print(f"Run on node {peer}: "+shlex.join(["bash","tests/ll_python_bench/run_all.sh",run_id,*tail]),flush=True)
    if a.dry_run:
        print(shlex.join(cmd)); return 0
    if not a.check_hccl:
        for variant in phase_names(a):
            directory = build.parent/variant
            if not list(directory.glob("_ll_py_combine*.so")):
                raise RuntimeError(f"Install {variant} into {sys.prefix}: bash tests/ll_python_bench/build_all.sh")
            if f"ranks_per_host={a.ranks_per_host}" not in (directory/"bench-build.txt").read_text().splitlines():
                raise RuntimeError(f"Rebuild {variant} with matching --ranks-per-host")
        if a.profile and not (build.parent/"bin/ll_python_profile_report").is_file():
            raise RuntimeError("Rebuild Python diagnostic variants: missing installed profile reporter")
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 65536: resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard) if hard>=0 else 65536,hard))
    node_dir = out / f"node-{node}"
    env = worker_environment(a,node_dir)
    network = configure_gloo_network(a,node,hosts,env)
    print(network,flush=True)
    node_dir.mkdir(parents=True,exist_ok=False)
    # Console is also saved locally. torchrun fails the local worker group on first error.
    with (node_dir/"console.log").open("w") as log:
        log.write(network+"\n"); log.flush()
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
