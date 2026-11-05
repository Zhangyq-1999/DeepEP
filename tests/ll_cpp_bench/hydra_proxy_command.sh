#!/usr/bin/env bash
# Hydra calls this local launcher to publish a command; launch.sh starts the
# real proxy on each host. No SSH connection or remote command is executed.
set -euo pipefail
while [[ ${1:-} == -x || ${1:-} == -X ]]; do shift; done
(($#>=2)) || { echo 'Missing Hydra host/proxy arguments' >&2; exit 2; }
host=$1 proxy=$2
shift 2
# MPICH quotes the executable when using its external-launcher path.
if [[ $proxy == \"*\" ]]; then proxy=${proxy:1:${#proxy}-2}; fi
args=("$@")
id= port=
while (($#>=2)); do
    case "$1" in --proxy-id) id=$2;; --control-port) port=$2;; esac
    shift 2
done
[[ $id =~ ^[0-9]+$ && $port =~ ^[a-zA-Z0-9_.-]+:[0-9]+$ && ${proxy##*/} == hydra_pmi_proxy ]] || {
    echo 'Invalid Hydra proxy command' >&2; exit 2;
}
printf -v proxy_line '%q ' "$proxy" "${args[@]}"
printf 'HYDRA_PROXY_COMMAND host=%s: env -u HYDI_CONTROL_FD %s\nLL_HYDRA_PROXY_READY %s %s\n' "$host" "$proxy_line" "$id" "$port"

# Hydra 4.1.3 treats launcher exit before the real proxy connects as a launch
# failure, even with exit status 0. Keep this local placeholder alive until
# launch.sh releases it after its real proxy exits (before waiting on Hydra).
# Parent death also releases it; no SSH or connection to the control port.
if [[ -n ${LL_HYDRA_LAUNCH_HOLD_FILE:-} ]]; then
    while [[ -f $LL_HYDRA_LAUNCH_HOLD_FILE ]] && kill -0 "$PPID" 2>/dev/null; do
        sleep 0.2
    done
fi
