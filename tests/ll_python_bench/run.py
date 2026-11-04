"""PyTorch synchronization around the unchanged C++ benchmark kernel and inputs."""
import argparse
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback


def stats(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Invalid Event samples")
    ordered = sorted(values)
    def percentile(p):
        x = p*(len(ordered)-1)
        lo = int(x)
        return ordered[lo]+(ordered[math.ceil(x)]-ordered[lo])*(x-lo)
    return dict(n=len(values),mean=statistics.mean(values),p50=percentile(.5),
                min=ordered[0],max=ordered[-1],p95=percentile(.95),p99=percentile(.99))


def summarize(per_rank):
    n = len(per_rank[0]) if per_rank else 0
    if not n or any(len(v)!=n for v in per_rank):
        raise ValueError("Missing rank/iteration samples")
    for v in per_rank: stats(v)
    maxima = [max(v[i] for v in per_rank) for i in range(n)]
    return dict(rank_max=stats(maxima),rank_max_tail50=stats(maxima[-50:]),
                rank0=stats(per_rank[0]),rank_max_us=maxima,per_rank_us=per_rank)


class DeviceGate:
    """Public HCCL all_reduce; mirrors Ziha0-Zeng/DeepEP b1ed463's gate."""
    def __init__(self, torch, dist, timeout, delay_ms=0):
        self.torch, self.dist = torch, dist
        self.group = dist.new_group(backend="hccl",timeout=timedelta(seconds=timeout))
        self.token = torch.ones(1,dtype=torch.float32,device="npu")
        self.delay_ms = 0
        self()
        torch.npu.synchronize()
        self.delay_ms = delay_ms

    def __call__(self):
        if self.delay_ms and self.dist.get_rank()==0: time.sleep(self.delay_ms/1000)
        work = self.dist.all_reduce(self.token,op=self.dist.ReduceOp.MAX,group=self.group,async_op=True)
        work.wait()
        # Submit torch_npu's queued completion dependency before direct ACL Events.
        _ = self.torch.npu.current_stream().npu_stream


class Progress:
    def __init__(self, enabled):
        self.enabled, self.stage, self.changed = enabled,"worker startup",time.monotonic()
        self.stopped = threading.Event()
        if enabled: threading.Thread(target=self.watch,daemon=True).start()

    def set(self, text):
        self.stage,self.changed = text,time.monotonic()
        if self.enabled: print(f"[{time.strftime('%H:%M:%S')}] {text}",flush=True)

    def watch(self):
        while not self.stopped.wait(15):
            print(f"[{time.strftime('%H:%M:%S')}] unchanged={time.monotonic()-self.changed:.0f}s {self.stage}",flush=True)


def native_args(a, local_size):
    args = ["--devices",",".join(map(str,range(local_size)))]
    for name in ("tokens","hidden","topk","experts","capacity","dtype","route","seed","warmup","iters","timeout","kernel_timeout"):
        args += ["--"+name.replace("_","-"),str(getattr(a,name))]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output","build","shmem-ip-port","variant","sync","devices","tokens","dtype","route"):
        p.add_argument("--"+name,required=True)
    for name in ("ranks-per-host","hidden","topk","experts","capacity","seed","warmup","iters","timeout","kernel-timeout"):
        p.add_argument("--"+name,type=int,required=True)
    p.add_argument("--gate-delay-ms",type=float,default=0)
    a = p.parse_args()
    rank,world,local = (int(os.environ[k]) for k in ("RANK","WORLD_SIZE","LOCAL_RANK"))
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    progress = Progress(local==0)
    progress.set(f"rank={rank} import Torch and initialize device")
    import torch
    import torch_npu
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.npu.set_device(local)
    torch.npu.init()
    sys.path.insert(0,a.build)
    import _ll_py_combine as native
    if native.ranks_per_host!=a.ranks_per_host or local_size!=a.ranks_per_host:
        raise ValueError("Python launch and compiled ranks-per-host differ")
    progress.set(f"rank={rank} Gloo rendezvous")
    dist.init_process_group("gloo",timeout=timedelta(seconds=a.timeout))
    # Agree before collective SHMEM initialization, even for different local paths.
    manifest = dict(line.split("=",1) for line in (Path(a.build)/"bench-build.txt").read_text().splitlines() if "=" in line)
    config = {k:v for k,v in vars(a).items() if k not in ("output","build")}
    config.update(source_id=manifest["source"],torch=torch.__version__,torch_npu=torch_npu.__version__,world=world)
    agreed = [None]*world
    dist.all_gather_object(agreed,config)
    if any(v!=agreed[0] for v in agreed): raise RuntimeError("Build/configuration differs across ranks")
    output = Path(a.output)
    output.mkdir(parents=True,exist_ok=True)
    rank_file = output/f"rank-{rank}.jsonl"
    stream = torch.npu.Stream()
    with rank_file.open("x") as raw, torch.npu.stream(stream):
        progress.set(f"rank={rank} SHMEM initialization (once)")
        bench = native.Bench(rank,world,local,native_args(a,local_size),stream.npu_stream,a.shmem_ip_port)
        dist.barrier()
        identity = bench.identity()
        dist.all_gather_object(agreed,identity)
        if any(v!=agreed[0] for v in agreed): raise RuntimeError("Native identity differs across ranks")
        gate = None
        if a.sync in ("device","both"):
            progress.set(f"rank={rank} HCCL gate initialization")
            gate = DeviceGate(torch,dist,a.timeout,a.gate_delay_ms)
        modes = ["host","device"] if a.sync=="both" else [a.sync]
        for bs in map(int,a.tokens.split(",")):
            progress.set(f"rank={rank} BS={bs} allocation and route preparation")
            dist.barrier(); bench.prepare(bs); dist.barrier()
            progress.set(f"rank={rank} BS={bs} Dispatch preparation")
            bench.dispatch(); dist.barrier(); bench.clear_flags(); dist.barrier()
            bench.sample(dist.barrier,False)
            error = bench.precision()
            dist.barrier()
            for mode in modes:
                callback = gate if mode=="device" else dist.barrier
                progress.set(f"rank={rank} BS={bs} sync={mode} warmup 0/{a.warmup}")
                for _ in range(a.warmup): bench.sample(callback,mode=="device")
                samples = []
                for i in range(a.iters):
                    if i%10==0: progress.set(f"rank={rank} BS={bs} sync={mode} samples {i}/{a.iters}")
                    samples.append(float(bench.sample(callback,mode=="device")))
                error = max(error,bench.precision())
                record = dict(config=config,identity=identity,rank=rank,tokens=bs,sync=mode,
                              measurement="native_kernel_event_us",max_abs_error=error,samples_us=samples)
                raw.write(json.dumps(record)+"\n"); raw.flush()
                progress.set(f"rank={rank} BS={bs} sync={mode} gather results")
                gathered = [None]*world
                dist.all_gather_object(gathered,dict(samples=samples,error=error))
                if rank==0:
                    result = summarize([v["samples"] for v in gathered])
                    result.update(config=config,identity=identity,tokens=bs,sync=mode,
                                  max_abs_error=max(v["error"] for v in gathered))
                    (output/f"bs{bs}-{mode}.json").write_text(json.dumps(result,indent=2)+"\n")
                    s = result["rank_max"]
                    path = output/"summary.csv"
                    exists = path.exists()
                    with path.open("a",newline="") as f:
                        row = dict(bs=bs,variant=a.variant,sync=mode,**s,
                                   tail50_p50=result["rank_max_tail50"]["p50"],rank0_p50=result["rank0"]["p50"],
                                   max_abs_error=result["max_abs_error"])
                        writer = csv.DictWriter(f,fieldnames=list(row))
                        if not exists: writer.writeheader()
                        writer.writerow(row)
                    print("\nBS variant sync N mean_us p50_us min_us p95_us max_us tail50_p50 rank0_p50 error",flush=True)
                    print(f"{bs} {a.variant} {mode} {s['n']} {s['mean']:.2f} {s['p50']:.2f} {s['min']:.2f} "
                          f"{s['p95']:.2f} {s['max']:.2f} {result['rank_max_tail50']['p50']:.2f} "
                          f"{result['rank0']['p50']:.2f} {result['max_abs_error']:.6g}\n",flush=True)
                dist.barrier()
        progress.set(f"rank={rank} SHMEM finalize")
        torch.npu.synchronize(); dist.barrier(); bench.close(); dist.barrier()
        (output/f"rank-{rank}.PASS").write_text("precision, sampling and SHMEM finalize passed\n")
        if gate is not None: dist.destroy_process_group(gate.group)
    dist.destroy_process_group()
    progress.set(f"rank={rank} PASS"); progress.stopped.set()


if __name__=="__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); sys.stderr.flush(); sys.stdout.flush()
        # Let torchrun terminate peers; don't enter SHMEM collectives or release
        # in-flight buffers while unwinding a failed distributed invocation.
        os._exit(1)
