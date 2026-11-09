# Combine source index width comparison

This is a benchmark-only experiment. The full ElasticBuffer Python API retains
its int32 Dispatch handle. No public handle ABI or Dispatch implementation is changed.

| Clean variant | Source top-k input | Read by Combine? | Comparison |
| --- | --- | --- | --- |
| input6-l2-route-only | int32 | no | Current optimized baseline |
| input6-l2-route-i64 | int64 | no | Same kernel schedule, true int64 argument storage |
| batch | int32 | yes | Existing local-first/vector final-route classification |
| batch-i64 | int64 | yes | Same classification using native int64 Vector Compares |

Each has a separate `-profile` build. Compare within each pair, not route-only
against batch: those use different reduction schedules. The default tiny-BS
suite is unchanged; these builds are opt-in.

Only `sourceIndices[B,K]` changes. Counts, destination/relay tables, source mask,
payload format, weights, WQEs, synchronization and QP/core ownership stay intact.
The benchmark generates identical routes, prepares Dispatch with int32, and
uploads the same values as int64 in case preparation. There is no device Cast
before or inside Combine. This tests accepting an int64 argument; it is not a
test of a future Dispatch implementation or total Dispatch+Combine latency.

The route-only pair deliberately has `source_indices_read=0`: its classification
comes from compressed Dispatch metadata. A fast result there does not prove
int64 loads/comparisons are free. The batch pair has `source_indices_read=1` and
performs real int64 DMA and Vector comparisons, without narrowing high bits.

## Build and run

For the current EP32 environment, run on both hosts:

```bash
git pull --ff-only
bash tests/ll_python_bench/build_all.sh --ranks-per-host 16 \
  --variants input6-l2-route-only,input6-l2-route-i64,batch,batch-i64
```

On host 0, then copy its printed command on host 1:

```bash
bash tests/ll_python_bench/run_all.sh --ranks-per-host 16 --sync hccl \
  --variants input6-l2-route-only,input6-l2-route-i64,batch,batch-i64
```

Python automatically appends all four diagnostic phases. Default cases remain
BS1,2,4,8,16,24. For the shortest acceptance-only check, select just
`input6-l2-route-only,input6-l2-route-i64` in both commands (four builds).
To measure actual int64 index processing only, select `batch,batch-i64` instead.
Omit `--ranks-per-host 16` for dual-host EP8 on devices 12–15.

C++ uses the same kernels and case generator. Select diagnostic names explicitly:

```bash
V=input6-l2-route-only,input6-l2-route-i64,batch,batch-i64,input6-l2-route-only-profile,input6-l2-route-i64-profile,batch-profile,batch-i64-profile
bash tests/ll_cpp_bench/build_all.sh --ranks-per-host 16 --variants "$V"
bash tests/ll_cpp_bench/run_all.sh --ep 32 --ranks-per-host 16 --variants "$V"
```

EP64 uses the existing four-host launcher and the same 16-ranks-per-host build.

## Evidence and limits

Clean Event mean/P50/min/max use the unchanged rank-MAX statistics. Host setup,
Dispatch and entry synchronization stay outside the Event. Int64 DMA and
comparisons in batch-i64 are inside it. Profile output keeps the existing
`batch_route_ticks` and `final_meta_load_ticks`; they are nested intervals, not
additive. The compact report's `batch group_checks/whole_tokens/route_ticks`
line includes batch route classification time. CSV/Perfetto exports remain enabled. The identity/summary
records `source_index_bits` and `source_indices_read` (Python ID: `srcidx`,
`srcidx_read`), and cross-rank plus Host/device checks reject mismatched widths.

UB snapshot capacity uses 8 bytes per int64 route and stays 256-byte aligned.
Existing flag scratch is reused before polling; M4 output lanes remain live and
separate. For H3584/K8/BS<=256 with 28 final owners, the existing flag snapshot
already covers the wider route data, so no input/output lane is removed.

Host checks execute production M5/M6 with delayed arrivals, BF16/FP16 bitwise
goldens, negative/high-word sentinels, partial outputs in flight and UB/output
guards. They cover 4/16 ranks per host and four host groups, plus script and
identity checks. CANN compilation and NPU performance still require device runs.

Ascend950's count-based Compares supports int64 in the upstream Ascend C API:
[Compares](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0068.html).
