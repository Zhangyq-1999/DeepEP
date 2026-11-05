#!/usr/bin/env bash
# One command per host; node 1 only needs the fresh run-id printed by node 0.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$root/tests/ll_cpp_bench/experiment_defaults.sh"
ep=8 ranks_per_host=4 node_rank=${NODE_RANK:-} hosts="${HOST0:-90.90.93.84},${HOST1:-90.90.93.184}"
master= port=29671 variants=$ll_bench_variants run_id= dry=0
cann=$ll_bench_cann mpi=$ll_bench_mpi shmem=$ll_bench_shmem
job_timeout=3600 join_timeout=300
bench_keys=(tokens hidden topk experts capacity dtype route seed entry-sync warmup iters profile-iters profile profile-frequency-hz timeout kernel-timeout)
declare -A bench=([tokens]=32,48,64,72,256 [hidden]=3584 [topk]=8 [experts]=384
                  [capacity]=256 [dtype]=bf16 [route]=random [seed]=0 [entry-sync]=device
                  [warmup]=10 [iters]=100 [profile-iters]=10 [profile]=2 [profile-frequency-hz]=1000000000 [timeout]=300 [kernel-timeout]=10)
fail() { echo "ERROR: $*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
Node 0: run_all.sh
Node 1: run_all.sh RUN_ID
Options: --ep 2|4|8|16|32|64 --ranks-per-host 4|16 --node-rank N --hosts HOST0,HOST1[,HOST2,HOST3]
         --master-addr IP --master-port PORT --run-id ID --variants NAME[,NAME...]
         --cann-home DIR --mpi-home DIR --shmem-root DIR --dry-run
         --join-timeout SECONDS --job-timeout SECONDS
Benchmark options (with or without a preceding --):
         --tokens LIST --devices LIST --hidden N --topk N --experts N --capacity N
         --dtype bf16|fp16 --route random|normal|hotspot --seed N --entry-sync device|host|auto
         --warmup N --iters N --profile-iters N --profile 0|1|2 --timeout SECONDS
         --kernel-timeout SECONDS (default 10; bounded stream wait)
         --profile-frequency-hz HZ (default 1000000000; 0 disables Perfetto export)
Defaults: EP8, four ranks/host, devices 12,13,14,15 on each host; hosts 90.90.93.84,90.90.93.184, port 29671.
          base/ready-snapshot/stage-stream, with matching diagnostics; BS 32,48,64,72,256; H3584/K8/E384/BF16.
          Warmup 10; clean samples 100; diagnostic samples 10, level 2.
Rebuild with the same --ranks-per-host value. Use 16 for EP16/32/64 on one/two/four hosts.
All selected variants share one MPI/ACL/SHMEM initialization.
Device MTE/RoCE barrier before each warmup/sample, outside Event timing. Use --entry-sync host for MPI comparison.
Node is detected from local addresses; --node-rank overrides detection.
Node 0 generates a new run-id on every invocation unless --run-id is supplied.
For custom hosts, ports or variants, use the command printed for each peer node.
Benchmark shapes and sampling options are propagated by MPI from node 0.
USAGE
            exit 0;;
        --dry-run) dry=1; shift;;
        --) shift;;
        --ep|--ranks-per-host|--node-rank|--hosts|--master-addr|--master-port|--variants|--run-id|--mpi-home|--cann-home|--shmem-root|--job-timeout|--join-timeout)
            (($#>=2)) || fail "Missing value: $1"
            key=$1 value=$2; shift 2
            case "$key" in
                --ep) ep=$value;; --ranks-per-host) ranks_per_host=$value;; --node-rank) node_rank=$value;; --hosts) hosts=$value;;
                --master-addr) master=$value;; --master-port) port=$value;; --variants) variants=$value;;
                --run-id) run_id=$value;; --mpi-home) mpi=$value;; --cann-home) cann=$value;;
                --shmem-root) shmem=$value;; --job-timeout) job_timeout=$value;; --join-timeout) join_timeout=$value;;
            esac;;
        --tokens|--devices|--hidden|--topk|--experts|--capacity|--dtype|--route|--seed|--entry-sync|--warmup|--iters|--profile-iters|--profile|--profile-frequency-hz|--timeout|--kernel-timeout)
            (($#>=2)) || fail "Missing value: $1"
            bench[${1#--}]=$2; shift 2;;
        -*) fail "Unknown option: $1";;
        *) [[ -z $run_id ]] || fail 'Specify only one run-id'; run_id=$1; shift;;
    esac
done
[[ $ep =~ ^(2|4|8|16|32|64)$ ]] || fail 'EP must be 2,4,8,16,32 or 64'
[[ $ranks_per_host =~ ^(4|16)$ ]] || fail 'ranks-per-host must be 4 or 16'
((ep<=16 || ranks_per_host==16)) || fail 'EP32/64 requires --ranks-per-host 16 in both build and run'
per_host=$((ep<ranks_per_host ? ep : ranks_per_host))
node_count=$((ep/per_host))
if [[ ! ${bench[devices]+set} ]]; then
    # Both current machines have 16 physical devices; use their last local ranks.
    bench[devices]=
    for ((device=16-per_host; device<16; ++device)); do
        bench[devices]+="${bench[devices]:+,}$device"
    done
fi
if ((ep>ranks_per_host)); then
    [[ $hosts =~ ^[a-zA-Z0-9_.-]+(,[a-zA-Z0-9_.-]+)*$ ]] || fail 'Provide comma-separated hostnames/IPv4 addresses'
    IFS=, read -r -a nodes <<< "$hosts"
    ((${#nodes[@]}==node_count)) || fail "EP=$ep with $per_host ranks/host requires $node_count hosts"
    for ((i=0;i<node_count;++i)); do
        for ((j=0;j<i;++j)); do
            [[ ${nodes[i]} != "${nodes[j]}" ]] || fail 'Each host must appear only once'
        done
    done
    master=${master:-${nodes[0]}}
    if [[ -z $node_rank ]]; then
        local_ids=$(hostname -I 2>/dev/null || true)
        local_ids+=" $(hostname 2>/dev/null || true) $(hostname -f 2>/dev/null || true)"
        if command -v ip >/dev/null; then
            local_ids+=" $(ip -o -4 addr show 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true)"
        fi
        for index in "${!nodes[@]}"; do
            for address in $local_ids; do
                if [[ $address == "${nodes[$index]}" ]]; then
                    [[ -z $node_rank || $node_rank == "$index" ]] || fail 'Multiple hosts match local addresses; specify --node-rank'
                    node_rank=$index
                fi
            done
        done
        [[ -n $node_rank ]] || fail "Cannot identify this host in $hosts; set --node-rank 0..$((node_count-1))"
    fi
else
    node_rank=${node_rank:-0}
    hosts=localhost
fi
[[ $node_rank =~ ^[0-3]$ ]] && ((node_rank<node_count)) || fail "node-rank must be in 0..$((node_count-1))"
[[ $ep -gt $ranks_per_host || $node_rank == 0 ]] || fail 'Single-host EP uses node-rank 0'
if [[ -z $run_id ]]; then
    [[ $node_rank == 0 ]] || fail 'Copy RUN_ID from node 0: bash tests/ll_cpp_bench/run_all.sh RUN_ID'
    run_id="ep${ep}-all-$(date +%Y%m%d-%H%M%S)-$$-$RANDOM"
fi
[[ $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail 'Use letters, digits, _, . or - for run-id'
[[ ${bench[profile]} =~ ^[012]$ ]] || fail 'profile must be 0, 1 or 2'
printf 'RUN_ID=%s\nnode=%s EP=%s variants=%s\n' "$run_id" "$node_rank" "$ep" "$variants"
if [[ $node_rank == 0 ]]; then
    printf 'BS=%s warmup=%s clean_iters=%s profile_iters=%s profile=%s entry_sync=%s route=%s seed=%s\n' "${bench[tokens]}" "${bench[warmup]}" "${bench[iters]}" "${bench[profile-iters]}" "${bench[profile]}" "${bench[entry-sync]}" "${bench[route]}" "${bench[seed]}"
else
    echo 'Benchmark shapes and sampling options come from node 0.'
fi
if [[ $ep -gt $ranks_per_host && $node_rank == 0 ]]; then
  for ((peer=1;peer<node_count;++peer)); do
    peer_cmd=(bash tests/ll_cpp_bench/run_all.sh "$run_id")
    # With defaults this prints only RUN_ID; custom launch settings travel too.
    if [[ $ep != 8 || $ranks_per_host != 4 || $hosts != 90.90.93.84,90.90.93.184 || $master != 90.90.93.84 || $port != 29671 || $variants != "$ll_bench_variants" ]]; then
        peer_cmd+=(--ep "$ep" --ranks-per-host "$ranks_per_host" --node-rank "$peer" --hosts "$hosts" --master-addr "$master" --master-port "$port" --variants "$variants")
    fi
    if [[ ${bench[profile]} != 2 ]]; then peer_cmd+=(--profile "${bench[profile]}"); fi
    if [[ $job_timeout != 3600 || $join_timeout != 300 ]]; then
        peer_cmd+=(--job-timeout "$job_timeout" --join-timeout "$join_timeout")
    fi
    if [[ $cann != "$ll_bench_cann" ]]; then peer_cmd+=(--cann-home "$cann"); fi
    if [[ $mpi != "$ll_bench_mpi" ]]; then peer_cmd+=(--mpi-home "$mpi"); fi
    if [[ $shmem != "$ll_bench_shmem" ]]; then peer_cmd+=(--shmem-root "$shmem"); fi
    printf 'Run on node %s: ' "$peer"; printf '%q ' "${peer_cmd[@]}"; printf '\n'
  done
fi
if ((!dry)); then ll_bench_environment "$cann" "$mpi" "$shmem"; fi
args=(--ep "$ep" --ranks-per-host "$ranks_per_host" --node-rank "$node_rank" --hosts "$hosts" --master-port "$port"
      --variants "$variants" --run-id "$run_id" --mpi-home "$mpi"
      --job-timeout "$job_timeout" --join-timeout "$join_timeout")
if [[ -n $master ]]; then args+=(--master-addr "$master"); fi
if ((dry)); then args+=(--dry-run); fi
args+=(--)
for key in "${bench_keys[@]}"; do args+=("--$key" "${bench[$key]}"); done
if [[ ${bench[devices]+set} ]]; then args+=(--devices "${bench[devices]}"); fi
exec bash "$root/tests/ll_cpp_bench/launch.sh" "${args[@]}"
