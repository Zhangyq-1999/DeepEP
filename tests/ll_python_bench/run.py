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
    worst = sorted(range(n),key=lambda i:maxima[i],reverse=True)[:3]
    rank_p50 = [stats(values)["p50"] for values in per_rank]
    return dict(rank_max=stats(maxima),rank_max_tail50=stats(maxima[-50:]),
                rank0=stats(per_rank[0]),rank_max_us=maxima,per_rank_us=per_rank,
                first_means={str(k):statistics.mean(maxima[:k]) for k in (3,10,20)},
                rank_p50_range=[min(rank_p50),max(rank_p50)],
                worst=[dict(iteration=i+1,rank=max(range(len(per_rank)),key=lambda r:per_rank[r][i]),us=maxima[i])
                       for i in worst])


TABLE_HEADER = " BS sync     N     mean      p50      min      p99       max   tail50  rank0p50 error"


def result_row(r):
    s = r["rank_max"]
    return (f"{r['tokens']:3} {r['sync']:6} {s['n']:3} {s['mean']:8.2f} {s['p50']:8.2f} {s['min']:8.2f} "
            f"{s['p99']:8.2f} {s['max']:9.2f} {r['rank_max_tail50']['p50']:8.2f} "
            f"{r['rank0']['p50']:9.2f} {r['max_abs_error']:.6g}")


def diagnostic_row(r):
    first = "/".join(f"{r['first_means'][str(k)]:.2f}" for k in (3,10,20))
    lo, hi = r["rank_p50_range"]
    worst = ",".join(f"i{w['iteration']}/r{w['rank']}:{w['us']:.2f}" for w in r["worst"])
    return (f"D BS={r['tokens']} sync={r['sync']} first3/10/20={first} "
            f"rankP50={lo:.2f}..{hi:.2f} worst={worst}")


def format_report(config, identity, results, run_id, status="RUNNING"):
    c = config
    lines = [f"LL_PY_REPORT_BEGIN RUN_ID={run_id} status={status}",
             f"CFG EP={c['world']} per_host={c['ranks_per_host']} devices={c['devices']} variant={c['variant']} "
             f"H={c['hidden']} K={c['topk']} E={c['experts']} dtype={c['dtype']} route={c['route']} seed={c['seed']} "
             f"cap={c['capacity']} warmup={c['warmup']} iters={c['iters']} gate_delay_ms={c['gate_delay_ms']}",
             f"SW source={c['source_id']} env={c['env']} python={c['python']} torch={c['torch']} npu={c['torch_npu']} "
             f"aiv={identity['active_aiv']} recv={identity['recv_cores']} qps={identity['rdma_qps']} "
             f"options={identity['return_options']}",
             "Event_us=per-iteration rank-MAX; no samples removed; worst iteration is 1-based", TABLE_HEADER]
    lines += [result_row(r) for r in results]
    lines += [diagnostic_row(r) for r in results]
    for bs in dict.fromkeys(r["tokens"] for r in results):
        modes = {r["sync"]:r["rank_max"]["p50"] for r in results if r["tokens"]==bs}
        if "host" in modes and "device" in modes:
            lines.append(f"AB BS={bs} device_p50_reduction={(1-modes['device']/modes['host'])*100:.2f}%")
    return "\n".join(lines+["LL_PY_REPORT_END"])+"\n"


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

    def set(self, text, emit=True):
        self.stage,self.changed = text,time.monotonic()
        if self.enabled and emit: print(f"[{time.strftime('%H:%M:%S')}] {text}",flush=True)

    def watch(self):
        while not self.stopped.wait(15):
            print(f"[{time.strftime('%H:%M:%S')}] unchanged={time.monotonic()-self.changed:.0f}s {self.stage}",flush=True)


def native_args(a, local_size):
    args = ["--devices",",".join(map(str,range(local_size)))]
    for name in ("tokens","hidden","topk","experts","capacity","dtype","route","seed","warmup","iters","timeout","kernel_timeout"):
        args += ["--"+name.replace("_","-"),str(getattr(a,name))]
    return args


class ConfigurationMismatch(RuntimeError):
    """The local leader already printed the shared mismatch report."""


def configuration_diff(records):
    lines = []
    for key in sorted(set().union(*(record.keys() for record in records))):
        groups = {}
        for rank, record in enumerate(records):
            value = json.dumps(record[key],sort_keys=True) if key in record else "<missing>"
            groups.setdefault(value,[]).append(str(rank))
        if len(groups)>1:
            lines.append(f"{key}: "+"; ".join(f"ranks={','.join(ranks)} value={value}" for value,ranks in groups.items()))
    return lines


def require_agreement(dist, record, world, rank, local, output, stage):
    agreed = [None]*world
    dist.all_gather_object(agreed,record)
    differences = configuration_diff(agreed)
    if local==0:
        output.mkdir(parents=True,exist_ok=True)
        (output/f"rank-{rank}-{stage}.json").write_text(json.dumps(agreed,indent=2)+"\n")
        if differences:
            print("\n".join([f"LL_PY_MISMATCH_BEGIN stage={stage}",*differences,"LL_PY_MISMATCH_END"]),flush=True)
    if differences:
        # Let every host save/print its diagnostics before torchrun stops peers.
        dist.barrier()
        raise ConfigurationMismatch(f"{stage} differs across ranks")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output","module-dir","shmem-ip-port","variant","sync","devices","tokens","dtype","route"):
        p.add_argument("--"+name,required=True)
    for name in ("ranks-per-host","hidden","topk","experts","capacity","seed","warmup","iters","timeout","kernel-timeout"):
        p.add_argument("--"+name,type=int,required=True)
    p.add_argument("--gate-delay-ms",type=float,default=0)
    a = p.parse_args()
    rank,world,local = (int(os.environ[k]) for k in ("RANK","WORLD_SIZE","LOCAL_RANK"))
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    progress = Progress(local==0)
    progress.set(f"rank={rank} import Torch")
    import torch
    import torch_npu
    import torch.distributed as dist
    torch.set_num_threads(1)
    progress.set(f"rank={rank} Gloo rendezvous")
    dist.init_process_group("gloo",timeout=timedelta(seconds=a.timeout))
    # Agree before collective SHMEM initialization, even for different local paths.
    manifest = dict(line.split("=",1) for line in (Path(a.module_dir)/"bench-build.txt").read_text().splitlines() if "=" in line)
    config = {k:v for k,v in vars(a).items() if k not in ("output","module_dir")}
    config.update(source_id=manifest["source"],env=Path(sys.prefix).name,python=sys.version.split()[0],
                  torch=torch.__version__,torch_npu=torch_npu.__version__,world=world)
    output = Path(a.output)
    progress.set(f"rank={rank} configuration check")
    require_agreement(dist,config,world,rank,local,output,"configuration")
    sys.path.insert(0,a.module_dir)
    import _ll_py_combine as native
    if native.ranks_per_host!=a.ranks_per_host or local_size!=a.ranks_per_host:
        raise ValueError("Python launch and compiled ranks-per-host differ")
    progress.set(f"rank={rank} initialize NPU")
    torch.npu.set_device(local)
    torch.npu.init()
    output.mkdir(parents=True,exist_ok=True)
    rank_file = output/f"rank-{rank}.jsonl"
    results = []
    stream = torch.npu.Stream()
    with rank_file.open("x") as raw, torch.npu.stream(stream):
        progress.set(f"rank={rank} SHMEM initialization (once)")
        bench = native.Bench(rank,world,local,native_args(a,local_size),stream.npu_stream,a.shmem_ip_port)
        dist.barrier()
        identity = bench.identity()
        require_agreement(dist,identity,world,rank,local,output,"native")
        gate = None
        if a.sync in ("device","both"):
            progress.set(f"rank={rank} HCCL gate initialization")
            gate = DeviceGate(torch,dist,a.timeout,a.gate_delay_ms)
        modes = ["host","device"] if a.sync=="both" else [a.sync]
        for bs in map(int,a.tokens.split(",")):
            progress.set(f"rank={rank} BS={bs} allocation and route preparation")
            dist.barrier(); bench.prepare(bs); dist.barrier()
            progress.set(f"rank={rank} BS={bs} Dispatch preparation",emit=False)
            bench.dispatch(); dist.barrier(); bench.clear_flags(); dist.barrier()
            progress.set(f"rank={rank} BS={bs} correctness",emit=False)
            bench.sample(dist.barrier,False)
            error = bench.precision()
            dist.barrier()
            for mode in modes:
                callback = gate if mode=="device" else dist.barrier
                progress.set(f"rank={rank} BS={bs} sync={mode} warmup 0/{a.warmup}",emit=False)
                for i in range(a.warmup):
                    bench.sample(callback,mode=="device")
                    progress.set(f"rank={rank} BS={bs} sync={mode} warmup {i+1}/{a.warmup}",emit=False)
                samples = []
                for i in range(a.iters):
                    progress.set(f"rank={rank} BS={bs} sync={mode} samples {i}/{a.iters}",emit=False)
                    samples.append(float(bench.sample(callback,mode=="device")))
                progress.set(f"rank={rank} BS={bs} sync={mode} precision check",emit=False)
                error = max(error,bench.precision())
                record = dict(config=config,identity=identity,rank=rank,tokens=bs,sync=mode,
                              measurement="native_kernel_event_us",max_abs_error=error,samples_us=samples)
                raw.write(json.dumps(record)+"\n"); raw.flush()
                progress.set(f"rank={rank} BS={bs} sync={mode} gather results",emit=False)
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
                    results.append(result)
                    (output/"report.txt").write_text(format_report(config,identity,results,output.name))
                    print("\n"+TABLE_HEADER+"\n"+result_row(result)+"\n"+diagnostic_row(result),flush=True)
                dist.barrier()
        progress.set(f"rank={rank} SHMEM finalize")
        torch.npu.synchronize(); dist.barrier(); bench.close(); dist.barrier()
        (output/f"rank-{rank}.PASS").write_text("precision, sampling and SHMEM finalize passed\n")
        if gate is not None: dist.destroy_process_group(gate.group)
    dist.destroy_process_group()
    if rank==0:
        report = format_report(config,identity,results,output.name,"PASS")
        (output/"report.txt").write_text(report)
        print("\n"+report,flush=True)
    progress.set(f"rank={rank} PASS"); progress.stopped.set()


if __name__=="__main__":
    try: main()
    except ConfigurationMismatch:
        os._exit(1)
    except BaseException:
        traceback.print_exc(); sys.stderr.flush(); sys.stdout.flush()
        # Let torchrun terminate peers; don't enter SHMEM collectives or release
        # in-flight buffers while unwinding a failed distributed invocation.
        os._exit(1)
