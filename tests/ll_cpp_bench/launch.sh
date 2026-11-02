#!/usr/bin/env bash
# One invocation per physical host; MPICH owns rank assignment and Host collectives.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
ep=16 node_rank=0 hosts=localhost master= port=29671 variants=base run_id=combine
job_timeout=3600 join_timeout=300 dry=0
bench_args=()
fail() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
launch.sh --ep 2|4|8|16|32 [--node-rank 0|1] [--hosts HOST0,HOST1]
          [--master-addr IP] [--master-port PORT] [--variants base|opt|base,opt]
          [--run-id NAME] [--mpi-home DIR] [--job-timeout SECONDS]
          [--join-timeout SECONDS] [--dry-run] -- [benchmark options]
Each host runs this script once. EP32 uses one MPICH 3.2.1 MPI world,
without SSH or manually copying proxy commands. Benchmark arguments come
from node 0. Both nodes use the same checkout path, hosts, ports, variants
and run-id. Results: results/ll-cpp/NAME/<phase>-<variant>/.
USAGE
            exit 0;;
        --dry-run) dry=1; shift;;
        --) shift; bench_args=("$@"); break;;
        --ep|--node-rank|--hosts|--master-addr|--master-port|--variants|--run-id|--mpi-home|--job-timeout|--join-timeout)
            (($#>=2)) || fail "Missing value: $1"
            key=$1 value=$2; shift 2
            case "$key" in
                --ep) ep=$value;; --node-rank) node_rank=$value;; --hosts) hosts=$value;;
                --master-addr) master=$value;; --master-port) port=$value;; --variants) variants=$value;;
                --run-id) run_id=$value;; --mpi-home) mpi=$value;; --job-timeout) job_timeout=$value;;
                --join-timeout) join_timeout=$value;;
            esac;;
        *) fail "Unknown launcher option: $1 (benchmark options follow --)";;
    esac
done
[[ $ep =~ ^(2|4|8|16|32)$ && $node_rank =~ ^[01]$ ]] || fail 'Invalid EP or node-rank'
[[ $hosts =~ ^[a-zA-Z0-9_.-]+(,[a-zA-Z0-9_.-]+)?$ ]] || fail 'Provide one or two hostnames/IPv4 addresses'
IFS=, read -r -a nodes <<< "$hosts"
if ((ep==32)); then
    ((${#nodes[@]}==2)) && [[ ${nodes[0]} != "${nodes[1]}" ]] || fail 'EP32 requires two distinct hosts'
    master=${master:-${nodes[0]}}
else
    ((node_rank==0 && ${#nodes[@]}==1)) || fail 'EP2/4/8/16 uses one host and node-rank 0'
fi
[[ -z $master || $master =~ ^[a-zA-Z0-9_.-]+$ ]] || fail 'Invalid master address'
[[ $variants =~ ^(base|opt)(,(base|opt))*$ ]] || fail 'Use base, opt, or a comma-separated sequence'
IFS=, read -r -a sequence <<< "$variants"
((${#sequence[@]}<=8)) || fail 'At most eight comparison phases are supported'
[[ $port =~ ^[1-9][0-9]{3,4}$ ]] && ((port>=1024 && port+${#sequence[@]}-1<=65535)) || fail 'Invalid master-port'
[[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail 'Use letters, digits, _, . or - for run-id'
[[ $job_timeout =~ ^[1-9][0-9]*$ && $join_timeout =~ ^[1-9][0-9]*$ ]] || fail 'Invalid timeout'
((join_timeout<=job_timeout && job_timeout<=86400)) || fail 'Require join-timeout <= job-timeout <= 86400'
for arg in "${bench_args[@]}"; do
    [[ $arg != --output-dir ]] || fail 'Use --run-id to select the output directory'
done
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$mpi/lib64:${SHMEM_ROOT:+$SHMEM_ROOT/lib:}${LD_LIBRARY_PATH:-}"
result_root="$root/results/ll-cpp/$run_id"
if ((!dry)); then
    [[ -x "$mpi/bin/mpiexec" ]] || fail 'MPI executable unavailable'
    if ((ep==32)); then
        [[ -x "$mpi/bin/hydra_pmi_proxy" ]] || fail 'Hydra proxy unavailable'
        "$mpi/bin/mpiexec" -version | grep -Eq 'Version:[[:space:]]+3\.2\.1([[:space:]]|$)' || fail 'Per-node startup requires MPICH 3.2.1'
    fi
    command -v setsid >/dev/null || fail 'setsid is required'
    command -v timeout >/dev/null || fail 'GNU timeout is required'
    for variant in "${sequence[@]}"; do
        [[ -x "$root/build/ll-cpp-$variant/ll_combine_bench" ]] || fail "Build ll-cpp-$variant first"
    done
    mkdir -p "$result_root"
    # Exclusive local ownership; node-specific directories also support shared storage.
    mkdir "$result_root/node-$node_rank" || fail 'This node already has this run-id; select a new one'
fi
controller= proxy=
cleanup() {
    local status=$?
    trap - EXIT INT TERM HUP
    # Signal only processes created by this invocation. Hydra SIGPIPE retires
    # its own application children; controller SIGINT cleans the MPI job.
    [[ -z $controller ]] || kill -INT "$controller" 2>/dev/null || true
    [[ -z $proxy ]] || kill -PIPE -- "-$proxy" 2>/dev/null || true
    if [[ -n $controller || -n $proxy ]]; then
        sleep 1
        [[ -z $controller ]] || kill -KILL -- "-$controller" 2>/dev/null || true
        [[ -z $proxy ]] || kill -KILL -- "-$proxy" 2>/dev/null || true
        [[ -z $controller ]] || wait "$controller" 2>/dev/null || true
        [[ -z $proxy ]] || wait "$proxy" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
print_cmd() { printf '%q ' "$@"; printf '\n'; }
for phase in "${!sequence[@]}"; do
    variant=${sequence[$phase]}
    phase_port=$((port+phase))
    out="$result_root/$phase-$variant"
    binary="$root/build/ll-cpp-$variant/ll_combine_bench"
    log="$result_root/node-$node_rank/$phase-$variant.log"
    echo "node=$node_rank EP=$ep phase=$phase variant=$variant output=$out"
    if ((ep<=16)); then
        cmd=("$mpi/bin/mpiexec" -launcher fork -rmk user -hosts localhost -ppn "$ep" -n "$ep"
             -bind-to none -wdir "$root" "$binary" "${bench_args[@]}" --output-dir "$out")
        if ((dry)); then print_cmd "${cmd[@]}"; continue; fi
        MPIEXEC_TIMEOUT=$job_timeout setsid "${cmd[@]}" >"$log" 2>&1 & controller=$!
        if wait "$controller"; then controller=; else status=$?; cat "$log"; exit "$status"; fi
    else
        if ((node_rank==0)); then
            # Only Hydra's controller port is fixed. MPI's data sockets retain
            # their normal allocation. Explicitly clear higher-priority aliases.
            cmd=(env)
            for name in MPIR_CVAR_CH3_PORT_RANGE MPIR_PARAM_CH3_PORT_RANGE MPICH_CH3_PORT_RANGE \
                        MPIR_CVAR_PORTRANGE MPIR_CVAR_PORT_RANGE MPIR_PARAM_PORTRANGE MPIR_PARAM_PORT_RANGE \
                        MPICH_PORTRANGE MPICH_PORT_RANGE MPIEXEC_PORTRANGE HYDI_CONTROL_FD; do
                cmd+=(-u "$name")
            done
            cmd+=("MPIEXEC_PORT_RANGE=$phase_port:$phase_port" "MPIEXEC_TIMEOUT=$job_timeout" HYDRA_LAUNCHER_AUTOFORK=0
                  "$mpi/bin/mpiexec" -launcher manual -rmk user -demux poll -localhost "$master"
                  -hosts "$hosts" -ppn 16 -n 32 -bind-to none -wdir "$root"
                  "$binary" "${bench_args[@]}" --output-dir "$out")
            if ((dry)); then print_cmd "${cmd[@]}"; else
                setsid "${cmd[@]}" >"$log.controller" 2>&1 & controller=$!
                # Wait for this controller to publish its proxy commands before
                # starting our proxy; fail locally if it could not bind the port.
                deadline=$((SECONDS+join_timeout))
                until grep -q '^HYDRA_LAUNCH_END' "$log.controller"; do
                    kill -0 "$controller" 2>/dev/null || { cat "$log.controller"; fail 'MPI controller exited'; }
                    ((SECONDS<deadline)) || fail 'MPI controller startup timed out'
                    sleep 0.1
                done
            fi
        fi
        cmd=(env -u HYDI_CONTROL_FD timeout --signal=PIPE --kill-after=5 "$job_timeout"
             "$mpi/bin/hydra_pmi_proxy" --control-port "$master:$phase_port" --rmk user
             --launcher manual --demux poll --pgid 0 --retries "$(((join_timeout+9)/10))"
             --usize "$ep" --proxy-id "$node_rank")
        if ((dry)); then print_cmd "${cmd[@]}"; continue; fi
        echo "Joining $master:$phase_port; log=$log"
        setsid "${cmd[@]}" >"$log" 2>&1 & proxy=$!
        if wait "$proxy"; then proxy=; else status=$?; cat "$log"; exit "$status"; fi
        if [[ -n $controller ]]; then
            if wait "$controller"; then controller=; else status=$?; cat "$log.controller"; exit "$status"; fi
            cat "$log.controller"
        fi
    fi
    cat "$log"
    per_host=$((ep<16 ? ep : 16))
    for ((rank=node_rank*per_host; rank<(node_rank+1)*per_host; ++rank)); do
        [[ -s "$out/rank-$rank.PASS" ]] || fail "Missing successful result for rank $rank; see $log"
    done
    echo "Completed phase $phase ($variant) on node $node_rank"
    [[ $node_rank != 0 ]] || cat "$out/summary.csv"
done
