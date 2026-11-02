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
printf 'HYDRA_PROXY_COMMAND host=%s: env -u HYDI_CONTROL_FD ' "$host"
printf '%q ' "$proxy" "${args[@]}"
printf '\nLL_HYDRA_PROXY_READY %s %s\n' "$id" "$port"
