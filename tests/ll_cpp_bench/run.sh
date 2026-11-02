#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mpi=${MPI_HOME:-/usr/local/mpich-3.2.1}
binary="$root/build/ll-cpp-base/ll_combine_bench"
hosts=localhost master= port_range=29671:29771 launcher=ssh job_timeout=3600 dry=0 ep=
bench_args=()
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'run.sh --hosts HOST0[,HOST1] [--ep 2|4|8|16|32] [--master-addr IP] [--port-range LOW:HIGH]'
            echo '       [--launcher ssh|manual|fork] [--mpi-home DIR] [--binary FILE] [--job-timeout SECONDS]'
            echo '       [--dry-run] -- [benchmark options; see binary --help]'
            exit 0;;
        --dry-run) dry=1;shift;;
        --) shift;bench_args=("$@");break;;
        --hosts|--ep|--master-addr|--port-range|--launcher|--mpi-home|--binary|--job-timeout)
            (($#>=2)) || { echo "Missing value: $1" >&2;exit 2; }
            key=$1 value=$2;shift 2
            case "$key" in
                --hosts) hosts=$value;; --ep) ep=$value;; --master-addr) master=$value;; --port-range) port_range=$value;;
                --launcher) launcher=$value;; --mpi-home) mpi=$value;; --binary) binary=$value;;
                --job-timeout) job_timeout=$value;;
            esac;;
        *) echo "Unknown launcher option: $1 (benchmark options follow --)" >&2;exit 2;;
    esac
done
[[ $hosts =~ ^[a-zA-Z0-9_.-]+(,[a-zA-Z0-9_.-]+)?$ ]] || { echo 'Provide one or two hostnames/IPv4 addresses' >&2;exit 2; }
IFS=, read -r -a nodes <<< "$hosts"
ep=${ep:-$((${#nodes[@]}*16))}
[[ $ep =~ ^(2|4|8|16|32)$ ]] || { echo 'EP must be 2, 4, 8, 16 or 32' >&2;exit 2; }
(( (${#nodes[@]}==1 && ep<=16) || (${#nodes[@]}==2 && ep==32) )) || { echo 'EP2/4/8/16 needs one host; EP32 needs two hosts' >&2;exit 2; }
ppn=$((ep/${#nodes[@]}))
if ((${#nodes[@]}==2)) && [[ ${nodes[0]} == "${nodes[1]}" ]]; then echo 'EP32 needs two physical hosts' >&2;exit 2;fi
[[ $launcher =~ ^(ssh|manual|fork)$ ]] || { echo 'Unknown launcher' >&2;exit 2; }
[[ $launcher != fork || ${#nodes[@]} == 1 ]] || { echo 'fork is only for a single host' >&2;exit 2; }
[[ $port_range =~ ^([0-9]+):([0-9]+)$ ]] || { echo 'Use --port-range LOW:HIGH' >&2;exit 2; }
low=${BASH_REMATCH[1]} high=${BASH_REMATCH[2]}
((10#$low>=1024 && 10#$high<=65535 && 10#$high>=10#$low)) || { echo 'Invalid port range' >&2;exit 2; }
[[ $job_timeout =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid job timeout' >&2;exit 2; }
[[ -z $master || $master =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'Invalid master address' >&2;exit 2; }
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$mpi/lib64:${SHMEM_ROOT:+$SHMEM_ROOT/lib:}${LD_LIBRARY_PATH:-}"
export MPIEXEC_PORT_RANGE="$port_range" MPIEXEC_TIMEOUT="$job_timeout"
# Manual mode prints one Hydra proxy command per host. Keep this mpiexec alive
# and run those commands on their respective hosts; all join one MPI world.
cmd=("$mpi/bin/mpiexec" -launcher "$launcher" -hosts "$hosts" -ppn "$ppn" -n "$ep"
     -bind-to none -wdir "$root")
[[ -z $master ]] || cmd+=(-localhost "$master")
cmd+=("$binary" "${bench_args[@]}")
if ((dry)); then
    printf 'MPIEXEC_PORT_RANGE=%q MPIEXEC_TIMEOUT=%q\n' "$port_range" "$job_timeout"
    printf '%q ' "${cmd[@]}";printf '\n';exit 0
fi
[[ -x "$mpi/bin/mpiexec" && -x "$binary" ]] || { echo 'Build the benchmark and select MPI first' >&2;exit 2; }
exec "${cmd[@]}"
