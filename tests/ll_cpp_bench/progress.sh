#!/usr/bin/env bash
# Host-only monitor. A repeated stage is a wait report, not proof of deadlock.
set -u
status_file=$1 node=$2 parent=$3 interval=${4:-15}
previous= last_change=$SECONDS last_print=$SECONDS
while kill -0 "$parent" 2>/dev/null; do
    current=
    if [[ -s $status_file ]]; then IFS= read -r current < "$status_file" || true; fi
    if [[ -n $current && $current != "$previous" ]]; then
        previous=$current last_change=$SECONDS last_print=$SECONDS
        printf '[%(%H:%M:%S)T] node=%s %s\n' -1 "$node" "$current"
    elif ((SECONDS-last_print>=interval)); then
        printf '[%(%H:%M:%S)T] node=%s unchanged=%ss %s\n' -1 "$node" "$((SECONDS-last_change))" \
            "${previous:-stage=MPI startup/rendezvous; waiting for benchmark process}"
        last_print=$SECONDS
    fi
    sleep 1
done
