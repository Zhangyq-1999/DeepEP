"""PyTorch synchronization around the unchanged C++ benchmark kernel and inputs."""
import argparse
import csv
import importlib.util
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
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


TABLE_HEADER = " BS variant                     kind    sync     N     mean      p50      min      p99       max   tail50  rank0p50 error"


def result_row(r):
    s = r["rank_max"]
    return (f"{r['tokens']:3} {r['variant']:27} {r.get('kind','clean'):7} {r['sync']:6} {s['n']:3} {s['mean']:8.2f} {s['p50']:8.2f} {s['min']:8.2f} "
            f"{s['p99']:8.2f} {s['max']:9.2f} {r['rank_max_tail50']['p50']:8.2f} "
            f"{r['rank0']['p50']:9.2f} {r['max_abs_error']:.6g}")


def diagnostic_row(r):
    first = "/".join(f"{r['first_means'][str(k)]:.2f}" for k in (3,10,20))
    lo, hi = r["rank_p50_range"]
    worst = ",".join(f"i{w['iteration']}/r{w['rank']}:{w['us']:.2f}" for w in r["worst"])
    return (f"D BS={r['tokens']} variant={r['variant']} sync={r['sync']} first3/10/20={first} "
            f"rankP50={lo:.2f}..{hi:.2f} worst={worst}")


def format_report(config, identity, results, run_id, status="RUNNING"):
    c = config
    lines = [f"LL_PY_REPORT_BEGIN RUN_ID={run_id} status={status}",
             f"CFG EP={c['world']} per_host={c['ranks_per_host']} devices={c['devices']} variant={c['variant']} "
             f"H={c['hidden']} K={c['topk']} E={c['experts']} dtype={c['dtype']} route={c['route']} seed={c['seed']} "
             f"cap={c['capacity']} warmup={c['warmup']} iters={c['iters']} gate_delay_ms={c['gate_delay_ms']}",
             f"SW source={c['source_id']} env={c['env']} python={c['python']} torch={c['torch']} npu={c['torch_npu']} "
             f"variants={','.join(identity)}",
             "SYNC host=Gloo device=SHMEM-mixed-71a2154 hccl=HCCL-all_reduce; device gates use local ACL Notify",
             "Event_us=per-iteration rank-MAX; no samples removed; worst iteration is 1-based",
             "clean=performance; profile=instrumented diagnostics; Perfetto uses independent core-entry origins", TABLE_HEADER]
    lines += [result_row(r) for r in results]
    lines += [(f"ID {v} aiv={i['active_aiv']} qps={i['rdma_qps']} layout={i['core_layout']} l2=upstream"
               if i.get("upstream_revision") else
               f"ID {v} aiv={i['active_aiv']} recv={i['recv_cores']} qps={i['rdma_qps']} options={i['return_options']} layout={i.get('core_layout','halves')} control_l2={int(i.get('control_l2',False))} data_l2={i.get('data_l2',0)} ready={i.get('ready_mode',0)} remote_first={int(i.get('remote_first',False))}")
              for v,i in identity.items()]
    lines += [f"REF {v} upstream={i['upstream_revision']} transport={i['transport']}; policy={i.get('ops_policy',0)} (0=reference,1=retire,2=overlap,3=device-CQ,4=batch-SQ+CQ; BS<=24 EP<=64); profile={int(i.get('profile_build',False))}"
              for v,i in identity.items() if i.get("upstream_revision")]
    lines += [diagnostic_row(r) for r in results]
    for variant,bs in dict.fromkeys((r["variant"],r["tokens"]) for r in results):
        modes = {r["sync"]:r["rank_max"]["p50"] for r in results if r["tokens"]==bs and r["variant"]==variant}
        if "host" in modes and "device" in modes:
            lines.append(f"AB BS={bs} variant={variant} device_p50_reduction={(1-modes['device']/modes['host'])*100:.2f}%")
    return "\n".join(lines+["LL_PY_REPORT_END"])+"\n"


class DeviceGate:
    """Public HCCL all_reduce; mirrors Ziha0-Zeng/DeepEP b1ed463's gate."""
    def __init__(self, torch, dist, timeout, delay_ms=0, progress=None):
        self.torch, self.dist = torch, dist
        report = progress or (lambda stage: None)
        report("HCCL group creation")
        self.group = dist.new_group(backend="hccl",timeout=timedelta(seconds=timeout+60))
        report("HCCL token allocation")
        self.token = torch.ones(1,dtype=torch.float32,device="npu")
        self.delay_ms = 0
        report("HCCL first all_reduce")
        self()
        report("HCCL initial stream synchronization")
        torch.npu.synchronize()
        self.delay_ms = delay_ms
        report("HCCL gate ready")

    def __call__(self):
        if self.delay_ms and self.dist.get_rank()==0: time.sleep(self.delay_ms/1000)
        work = self.dist.all_reduce(self.token,op=self.dist.ReduceOp.MAX,group=self.group,async_op=True)
        work.wait()
        # Submit torch_npu's queued completion dependency before direct ACL Events.
        _ = self.torch.npu.current_stream().npu_stream


class ShmemGate:
    """Same mixed MTE/RoCE barrier kernel as the C++ benchmark; no HCCL group."""
    def __init__(self, bench, dist, delay_ms=0):
        self.bench, self.dist, self.delay_ms = bench, dist, delay_ms
        bench.check_entry_sync()

    def __call__(self):
        if self.delay_ms and self.dist.get_rank()==0: time.sleep(self.delay_ms/1000)
        self.bench.entry_sync()


def sync_implementation(mode):
    return {"host":"gloo", "device":"shmem-mixed-71a2154", "hccl":"hccl-all_reduce"}[mode]


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


def native_args(a):
    args = ["--devices",a.devices]
    for name in ("tokens","hidden","topk","experts","capacity","dtype","route","seed","warmup","iters","timeout","kernel_timeout",
                 "profile","profile_iters","profile_frequency_hz"):
        args += ["--"+name.replace("_","-"),str(getattr(a,name))]
    return args


class ConfigurationMismatch(RuntimeError):
    """The local leader already printed the shared mismatch report."""


def configuration_diff(records, allow_python_patch=False):
    lines = []
    for key in sorted(set().union(*(record.keys() for record in records))):
        # Each host uses its own installed extension. A CPython patch release
        # difference alone does not make the distributed test incompatible.
        if key=="python" and allow_python_patch and all(key in record for record in records):
            if len({tuple(record[key].split(".")[:2]) for record in records})==1:
                continue
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
    differences = configuration_diff(agreed,allow_python_patch=stage=="configuration")
    if local==0:
        output.mkdir(parents=True,exist_ok=True)
        (output/f"rank-{rank}-{stage}.json").write_text(json.dumps(agreed,indent=2)+"\n")
        if differences:
            print("\n".join([f"LL_PY_MISMATCH_BEGIN stage={stage}",*differences,"LL_PY_MISMATCH_END"]),flush=True)
        elif stage=="configuration":
            patch_diff = configuration_diff([dict(python=v["python"]) for v in agreed])
            if patch_diff: print("INFO allowed Python patch difference: "+patch_diff[0],flush=True)
    if differences:
        # Let every host save/print its diagnostics before torchrun stops peers.
        dist.barrier()
        raise ConfigurationMismatch(f"{stage} differs across ranks")


def load_native(directory, variant):
    paths = list(directory.glob("_ll_py_combine*.so"))
    if len(paths)!=1:
        raise RuntimeError(f"Rebuild Python variant {variant}: expected one extension in {directory}")
    # Same PyInit name, distinct module objects and module-local pybind classes.
    name = "_ll_variant_"+variant.replace("-","_")+"._ll_py_combine"
    spec = importlib.util.spec_from_file_location(name,paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    if getattr(module,"suite_api",0)!=3:
        raise RuntimeError(f"Rebuild {variant}: bash tests/ll_python_bench/build_all.sh")
    return module


def append_csv(path, rows):
    rows = list(rows)
    if not rows: return
    exists = path.exists()
    with path.open("a",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists: writer.writeheader()
        writer.writerows(rows)


def write_timing_artifacts(directory, result):
    """Same offline report format as C++; only global rank 0 writes these files."""
    directory.mkdir(parents=True,exist_ok=True)
    c,i,s = result["config"],result["identity"],result["rank_max"]
    append_csv(directory/"summary.csv",[dict(
        ep=c["world"],tokens=result["tokens"],hidden=c["hidden"],topk=c["topk"],experts=c["experts"],
        dtype=c["dtype"],route=c["route"],seed=c["seed"],capacity=c["capacity"],warmup=c["warmup"],
        source_id=i["source_id"],samples=s["n"],active_aiv=i["active_aiv"],recv_cores=i["recv_cores"],
        entry_sync=result["sync"],profile_build=int(result["kind"]=="profile"),
        profile_level=c["profile"] if result["kind"]=="profile" else 0,
        rank_max_mean_us=s["mean"],rank_max_min_us=s["min"],rank_max_max_us=s["max"],p50_us=s["p50"],p99_us=s["p99"],
        core_layout=i.get("core_layout","halves"),control_l2=int(i.get("control_l2",False)),data_l2=i.get("data_l2",0),
        ready=i.get("ready_mode",0),remote_first=int(i.get("remote_first",False)),
        upstream_revision=i.get("upstream_revision",""),transport=i.get("transport",""),ops_policy=i.get("ops_policy",0))])
    ranks = result["per_rank_us"]
    append_csv(directory/"rank-max.csv",[dict(tokens=result["tokens"],iteration=j,
        rank=max(range(len(ranks)),key=lambda r:ranks[r][j]),us=value)
        for j,value in enumerate(result["rank_max_us"])])


def gather_profile_bytes(dist, world, value):
    parts = [None]*world
    dist.all_gather_object(parts,value)
    if any(not isinstance(p,bytes) or len(p)!=len(value) for p in parts):
        raise ValueError("Incomplete native profile gather")
    return b"".join(parts)


def render_profiles(module_root, output, modes):
    reporter = module_root/"bin/ll_python_profile_report"
    for mode in modes:
        result = subprocess.run([str(reporter),"--suite",str(output/"profiles"/mode)],
                                text=True,capture_output=True,check=True)
        (output/f"profiling-{mode}.txt").write_text(result.stdout)
        print(f"LL_PY_PROFILE sync={mode}\n"+result.stdout,flush=True)
    if any((output/"profiles").glob("*/*/perfetto/*.perfetto.json")):
        print(f"Perfetto: {output}/profiles/<sync>/<phase>-<variant>/perfetto/bs*-*.perfetto.json",flush=True)


def runtime_diagnostics(output, rank, device, stage, local):
    keys = ("ASCEND_HOME_PATH","ASCEND_OPP_PATH","ASCEND_AICPU_PATH","LD_LIBRARY_PATH",
            "HCCL_OP_EXPANSION_MODE","HCCL_ALGO","HCCL_SOCKET_IFNAME","GLOO_SOCKET_IFNAME",
            "ASCEND_RT_VISIBLE_DEVICES","TORCH_HCCL_BLOCKING_WAIT")
    maps = Path("/proc/self/maps")
    libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                        if any(name in line for name in ("libascendcl", "libruntime", "libhccl", "libhcomm", "libtorch_npu"))}) if maps.exists() else []
    info = dict(stage=stage,rank=rank,device=device,pid=os.getpid(),python=sys.executable,
                env={key:os.environ.get(key,"") for key in keys},libraries=libraries)
    (output/f"rank-{rank}-runtime.json").write_text(json.dumps(info,indent=2)+"\n")
    if local==0:
        print(f"RUNTIME rank={rank} device={device} CANN={os.environ.get('ASCEND_HOME_PATH','unset')} "
              f"AICPU={os.environ.get('ASCEND_AICPU_PATH','unset')}",flush=True)
        for library in libraries: print("LIB "+library,flush=True)


def initialize_hccl(torch, dist, a, progress, output, rank, local, device, stage):
    progress.set(f"rank={rank} HCCL check ({stage})")
    runtime_diagnostics(output,rank,device,stage,local)
    try:
        gate = DeviceGate(torch,dist,a.timeout,a.gate_delay_ms,
                          lambda text: progress.set(f"rank={rank} {text}"))
    except Exception:
        runtime_diagnostics(output,rank,device,stage+" failed",local)
        print(f"HCCL_INIT_FAIL rank={rank} device={device} stage={stage}; Combine not started",flush=True)
        raise
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output","module-dir","shmem-ip-port","variant","sync","devices","tokens","dtype","route"):
        p.add_argument("--"+name,required=True)
    for name in ("ranks-per-host","hidden","topk","experts","capacity","seed","warmup","iters","timeout","kernel-timeout"):
        p.add_argument("--"+name,type=int,required=True)
    p.add_argument("--gate-delay-ms",type=float,default=0)
    p.add_argument("--variants",help="Clean and diagnostic phases sharing one SHMEM session")
    p.add_argument("--profile",type=int,choices=(0,1,2),default=0)
    p.add_argument("--profile-iters",type=int,default=10)
    p.add_argument("--profile-frequency-hz",type=int,default=1000000000)
    p.add_argument("--check-hccl",action="store_true")
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
    names = a.variants.split(",") if a.variants else [a.variant]
    directories = {name:Path(a.module_dir).parent/name for name in names}
    directories[names[0]] = Path(a.module_dir)
    sources = {} if a.check_hccl else {
        name:dict(line.split("=",1) for line in (directory/"bench-build.txt").read_text().splitlines() if "=" in line)["source"]
        for name,directory in directories.items()}
    config = {k:v for k,v in vars(a).items() if k not in ("output","module_dir")}
    config.update(source_id=sources,variant=",".join(names),env=Path(sys.prefix).name,python=sys.version.split()[0],
                  torch=torch.__version__,torch_npu=torch_npu.__version__,world=world,
                  device_sync_impl="shmem-mixed-71a2154")
    output = Path(a.output)
    progress.set(f"rank={rank} configuration check")
    require_agreement(dist,config,world,rank,local,output,"configuration")
    if local_size!=a.ranks_per_host:
        raise ValueError("Python launch and compiled ranks-per-host differ")
    device = int(a.devices.split(",")[local])
    progress.set(f"rank={rank} initialize NPU device={device}")
    torch.npu.set_device(device)
    torch.npu.init()
    actual = torch.npu.current_device()
    if actual!=device: raise RuntimeError(f"rank={rank} requested device={device}, current={actual}")
    mapping = [None]*world
    dist.all_gather_object(mapping,dict(rank=rank,local_rank=local,device=actual))
    if local==0:
        print("BIND "+" ".join(f"r{v['rank']}=dev{v['device']}" for v in mapping[rank:rank+local_size]),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    gate = None
    if a.sync=="hccl" or a.check_hccl:
        # HCCL initializes its Device runtime before SHMEM/Hcomm claims resources.
        # Do not even load the native extension before the first all_reduce.
        gate = initialize_hccl(torch,dist,a,progress,output,rank,local,device,"before native/SHMEM")
        dist.barrier()
    if a.check_hccl:
        dist.barrier()
        for _ in range(3): gate(); torch.npu.synchronize()
        dist.barrier(); dist.destroy_process_group(gate.group); dist.destroy_process_group()
        (output/f"rank-{rank}.HCCL_PASS").write_text("Torch-only HCCL check passed\n")
        if local==0: print(f"HCCL_CHECK_PASS rank={rank} devices={a.devices}; no Combine/SHMEM loaded",flush=True)
        progress.stopped.set()
        return
    modules = {name:load_native(directory,name) for name,directory in directories.items()}
    if any(m.ranks_per_host!=a.ranks_per_host for m in modules.values()) or len(set(sources.values()))!=1:
        raise ValueError("Rebuild all suite variants with matching source and ranks-per-host")
    rank_file = output/f"rank-{rank}.jsonl"
    results = []
    identities = {}
    stream = torch.npu.Stream()
    with rank_file.open("x") as raw, torch.npu.stream(stream):
        progress.set(f"rank={rank} SHMEM initialization (once)")
        owner = modules[names[0]].Bench(rank,world,local,native_args(a),stream.npu_stream,a.shmem_ip_port)
        dist.barrier()
        if a.sync=="hccl":
            progress.set(f"rank={rank} verify existing HCCL gate after SHMEM initialization")
            gate(); torch.npu.synchronize(); dist.barrier()
        modes = ["host","device"] if a.sync=="both" else [a.sync]
        for phase,variant in enumerate(names):
            progress.set(f"rank={rank} phase={phase} variant={variant}")
            torch.npu.synchronize(); dist.barrier()
            if phase: owner.reset_workspace()
            dist.barrier()
            bench = owner if phase==0 else modules[variant].Bench(owner.share())
            phase_output = output/variant
            phase_output.mkdir(parents=True,exist_ok=True)
            identity = bench.identity()
            require_agreement(dist,identity,world,rank,local,phase_output,"native")
            is_profile = bool(identity.get("profile_build",False))
            if is_profile != variant.endswith("-profile"):
                raise RuntimeError(f"Rebuild {variant}: clean/profile identity mismatch")
            iterations = a.profile_iters if is_profile else a.iters
            identities[variant] = identity
            if rank==0 and is_profile: print("\n"+TABLE_HEADER,flush=True)
            if a.sync in ("device","both"):
                progress.set(f"rank={rank} SHMEM device entry sync preparation")
                gate = ShmemGate(bench,dist,a.gate_delay_ms)
            for bs in map(int,a.tokens.split(",")):
                progress.set(f"rank={rank} variant={variant} BS={bs} allocation and route preparation")
                dist.barrier(); bench.prepare(bs); dist.barrier()
                progress.set(f"rank={rank} variant={variant} BS={bs} Dispatch preparation",emit=False)
                bench.dispatch(); dist.barrier(); bench.clear_flags(); dist.barrier()
                progress.set(f"rank={rank} variant={variant} BS={bs} correctness",emit=False)
                bench.sample(dist.barrier,False)
                error = bench.precision()
                dist.barrier()
                progress.set(f"rank={rank} variant={variant} BS={bs} cache reuse correctness",emit=False)
                # Each bank sees -x then +x at the same addresses. Do not clear
                # flags between generations; original input is restored before warmup.
                for generation in range(4):
                    if generation % 2 == 0: bench.flip_inputs()
                    dist.barrier()
                    bench.sample(dist.barrier,False)
                    error = max(error,bench.precision())
                    dist.barrier()
                for mode in modes:
                    callback = dist.barrier if mode=="host" else gate
                    progress.set(f"rank={rank} variant={variant} BS={bs} sync={mode} warmup 0/{a.warmup}",emit=False)
                    for i in range(a.warmup):
                        bench.sample(callback,mode!="host")
                        progress.set(f"rank={rank} variant={variant} BS={bs} sync={mode} warmup {i+1}/{a.warmup}",emit=False)
                    samples = []
                    diagnostics = output/"profiles"/mode/f"{phase}-{variant}"
                    if is_profile: bench.begin_profile(str(diagnostics),mode)
                    for i in range(iterations):
                        progress.set(f"rank={rank} variant={variant} BS={bs} sync={mode} samples {i}/{iterations}",emit=False)
                        samples.append(float(bench.sample(callback,mode!="host")))
                    progress.set(f"rank={rank} variant={variant} BS={bs} sync={mode} precision check",emit=False)
                    error = max(error,bench.precision())
                    record = dict(config=config,identity=identity,rank=rank,variant=variant,tokens=bs,sync=mode,
                                  sync_impl=sync_implementation(mode),
                                  kind="profile" if is_profile else "clean",
                                  measurement="native_kernel_event_us",max_abs_error=error,samples_us=samples)
                    raw.write(json.dumps(record)+"\n"); raw.flush()
                    progress.set(f"rank={rank} variant={variant} BS={bs} sync={mode} gather results",emit=False)
                    gathered = [None]*world
                    dist.all_gather_object(gathered,dict(samples=samples,error=error))
                    if is_profile:
                        progress.set(f"rank={rank} variant={variant} BS={bs} sync={mode} profile gather/Perfetto",emit=False)
                        maxima = [max(v["samples"][j] for v in gathered) for j in range(iterations)]
                        bench.finish_profile(maxima,lambda data:gather_profile_bytes(dist,world,data))
                    if rank==0:
                        result = summarize([v["samples"] for v in gathered])
                        result.update(config=config,identity=identity,variant=variant,tokens=bs,sync=mode,
                                      kind="profile" if is_profile else "clean",
                                      sync_impl=sync_implementation(mode),
                                      max_abs_error=max(v["error"] for v in gathered))
                        (phase_output/f"bs{bs}-{mode}.json").write_text(json.dumps(result,indent=2)+"\n")
                        write_timing_artifacts(diagnostics,result)
                        s = result["rank_max"]
                        path = output/"summary.csv"
                        exists = path.exists()
                        with path.open("a",newline="") as f:
                            row = dict(bs=bs,variant=variant,kind=result["kind"],sync=mode,sync_impl=sync_implementation(mode),**s,
                                       tail50_p50=result["rank_max_tail50"]["p50"],rank0_p50=result["rank0"]["p50"],
                                       max_abs_error=result["max_abs_error"])
                            writer = csv.DictWriter(f,fieldnames=list(row))
                            if not exists: writer.writeheader()
                            writer.writerow(row)
                        results.append(result)
                        (output/"report.txt").write_text(format_report(config,identities,results,output.name))
                        # Profile distribution details are in the final compact report.
                        # Keep interim progress small enough to paste with that report.
                        print(result_row(result) if is_profile else
                              "\n"+TABLE_HEADER+"\n"+result_row(result)+"\n"+diagnostic_row(result),flush=True)
                    dist.barrier()
            torch.npu.synchronize(); dist.barrier()
            if phase==0: owner.release_case()
            else: bench.close()
            dist.barrier()
            (phase_output/f"rank-{rank}.SAMPLED").write_text("precision and sampling passed\n")
        progress.set(f"rank={rank} SHMEM finalize")
        torch.npu.synchronize(); dist.barrier(); owner.close(); dist.barrier()
        if a.sync=="hccl": dist.destroy_process_group(gate.group)
    if rank==0 and any(i.get("profile_build") for i in identities.values()):
        progress.set("rank=0 render compact profile report")
        render_profiles(Path(a.module_dir).parent,output,modes)
    dist.barrier()
    (output/f"rank-{rank}.PASS").write_text("precision, sampling, profiling and SHMEM finalize passed\n")
    dist.destroy_process_group()
    if rank==0:
        report = format_report(config,identities,results,output.name,"PASS")
        (output/"report.txt").write_text(report)
        if any(i.get("profile_build") for i in identities.values()):
            # The compact report already includes clean/profile tables for each BS.
            print("\n"+"\n".join(line for line in report.splitlines()
                if line.startswith(("LL_PY_REPORT_","CFG ","SW ","SYNC ","ID ","REF "))),flush=True)
        else: print("\n"+report,flush=True)
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
