# M4 transfer experiments

Target: EP8/EP32 BS32/48/64, with BS72 and BS256 retained. Experiments use
`ready-snapshot`'s route planning, dependency scheduler and 28/28 core split.
The frozen `base` and existing Dispatch sources are unchanged.

## wqe-prep (ready=9)

The per-core staging address is known before the payload copy completes.
After queuing that copy and MTE3_S(7), construct a single HNS-1825 WRITE WQE
in an independent 128-byte UB area. The public SDK encoder supplies control,
task and data fields, including owner/wrap, remote MR translation and keys.
No valid SQ bytes or producer index are published yet. After the payload
wait, use the SDK's next-owner invalidation, UB-to-SQ copy, cache flush and
doorbell helper. Each peer/core still owns its original QP and generates one
WQE and one CQE per packet. The SDK doorbell updates its normal head mirror.

Admission uses min(SQ depth, CQ depth) and the SDK's ten-entry margin. A full
queue cannot poll CQ while the payload event is live; that packet falls back
to the unchanged SDK put after the payload wait. No queue reservation crosses
another submit, dependency wait, kernel or core. This adapter targets the
locked HNS-1825 implementation (`4c671684e1698aec199619850a3120bfdf85c95e`),
not XSCALE aggregate APIs. The SHMEM submodule and installed SDK are not edited.

The private UB is included in M4 workspace accounting and cannot overlap
inputs, outputs or the reduced `low_` buffer. Planning may borrow idle payload
space before use. The descriptor and doorbell copy finish before the next
packet or M5 reuses phase scratch. DIRECT cores do not enable preparation.

## mte-pack (ready=10)

Keep the final FP32-to-BF16/FP16 conversion and arithmetic order unchanged.
For remote partials, omit the two Vector copies that interleave 480-byte
payload blocks with 32-byte flags in the output UB. Instead issue:

1. Strided payload UB-to-GM copies for complete 480-byte blocks.
2. A final short payload copy if required.
3. Strided copies of the original initialized 32-byte flags into that GM packet.

UB source stride uses 32-byte units; GM destination stride uses bytes. The
packet length, every flag position and all valid hidden elements match the
original wire. Unused padding in the last payload block is not consumed.
The original V-to-MTE3 dependency is retained. The immediate MTE3_S wait
protects `low_` before a subsequent reduction can overwrite it. Only after
all payload and flag copies finish may the SDK borrow the packet's first
128 bytes as scratch and publish the RDMA write. Local returns use the old
packed output path. No additional UB or GM is allocated by this variant.

Both experiments preserve all 128 KiB/core GM slots, one quiet per touched
peer at a capacity boundary, and the final flush after M5. M5 never overwrites
the GM staging source. Two- and four-host topologies use the existing physical
rank grouping; there is no assumption of exactly one remote host.

## wqe-prefetch (ready=11)

Start from wqe-prep. When another task in the SAME already-ready selection
follows and there are at least two output lanes, keep the current payload's
MTE3 event and private UB WQE live. Load the next task's metadata and issue its
first up-to-three MTE2 input reads, then wait for the previous MTE3 and publish
its WQE BEFORE computing the next partial. The current input reads can progress
while Scalar waits for payload staging and submits the RDMA write.

Unlike stage-stream, this does not wait for the next full reduction/packing.
Unlike immediate wqe-prep, the first packet may be delayed by the next input
issues; its first/last Put timestamps and clean latency must be checked. A
single task, a last-ready task, a single output lane, and DIRECT keep immediate
submission. There is no pending WQE across a group-ready poll or M5 entry.

Only same-host MTE2 input reads occur between Prepare and Publish; no other
writer can advance this core/peer SQ head in that interval. The registered GM
source survives until quiet. The private 128-byte descriptor, prior output,
next inputs, metadata and reduction buffers are disjoint. No extra UB or GM
compared with wqe-prep, and no arithmetic, flags, packet count, queue owner or
physical-host grouping changes. This applies to both two- and four-host cases.

The hypothesis is that advancing independent input reads reduces exposed
M4 wait time at BS32/48/64. A single packet per receive core cannot benefit;
metadata/issue cost or MTE2/MTE3 contention can instead make it slower.

## Read the experiment

Default C++ build/run includes base, ready-snapshot, wqe-prep and wqe-prefetch,
with four matching profile variants. mte-pack remains opt-in.
Compare clean mean, P50 and min/max first. In diagnostics, compare M4 pack,
stage issue, remaining stage wait, put, first/last put, final flag wait and
quiet for the same rank/core/sample. `xfer` appends prepare ticks, prepared
packets, SDK fallbacks and direct MTE packets; full CSV/Perfetto retains them.
For wqe-prefetch, `rdma` includes deferred packet count and elapsed defer gap;
`stage` includes input-prefetch submit count. Those existing fields also reach
the CSV and Perfetto counters, with no extra clean-kernel timestamps.
Preparation overlaps DMA, so its elapsed ticks are not additive network time.
Extra MTE commands or Scalar preparation may cost more than they save. None
of these experiments reduces the number of network bytes or WQEs.

The first EP8 run at `f9eec20` did not improve clean latency over
`ready-snapshot`. It also exposed a shared retirement regression: TransferBatch
scanned every peer bit, even with no outstanding requests or an empty mask word.
For the printed BS32/48/64 P50-representative receive cores,
`FlushDone - PipeDone - RdmaQuietTicks` was 2555--2748 ticks for the experiments
versus 516--598 ticks for `ready-snapshot`.
Retirement now restores the original empty-batch return and empty-word skip.
This preserves every nonempty peer's quiet and GM lifetime, including EP64
peers in the second mask word. The isolated timing evidence identifies avoidable
bookkeeping rather than a network improvement.

## EP32 retirement-fix result and next comparison

User-provided run after 339977a: H3584/K8/E384/BF16, random seed 0, device
entry sync, warmup 10, 100 clean samples; global rank-MAX each iteration.
No samples removed. Values below are mean / P50 / min..max in microseconds.

| BS | base | ready-snapshot | wqe-prep | mte-pack |
|---:|---|---|---|---|
|32|77.64 / 77.74 / 62.13..98.18|69.91 / 68.59 / 57.76..86.64|70.26 / 69.01 / 57.18..87.45|70.29 / 68.04 / 57.90..89.98|
|48|82.94 / 82.91 / 66.84..102.61|78.06 / 77.27 / 65.17..95.06|76.34 / 73.76 / 65.31..92.73|79.26 / 79.49 / 64.39..96.01|
|64|91.53 / 90.49 / 77.69..110.68|86.29 / 84.31 / 74.59..102.89|83.84 / 82.35 / 73.11..98.51|85.14 / 83.41 / 73.71..102.98|
|72|94.00 / 92.54 / 80.90..111.57|89.77 / 89.02 / 77.48..108.40|88.34 / 86.44 / 76.17..105.01|90.14 / 90.14 / 77.85..105.71|
|256|200.82 / 201.04 / 191.50..215.05|193.83 / 193.16 / 185.54..206.69|191.74 / 190.83 / 184.53..202.86|193.47 / 192.58 / 185.80..207.65|

For BS32/48/64 representative receive cores, non-quiet retirement fell from
2.796/2.748/2.808 us to 0.593/0.618/0.698 us in wqe-prep. Quiet remains intact.
BS48/64 clean means improve 2.20%/2.84% over snapshot within this run; BS32
is flat. Run-to-run baseline movement prevents attributing all deltas to the
fix, and 7--12% small-BS CV calls for repeated measurements.

The BS64 wqe-prep representative core sends three packets: prepare=1921,
stage_wait=15689, put=2343 ticks, no fallback. This motivates input prefetch,
not a claim that network transit costs 15689 ticks. Missing final flags are
mostly RDMA partials, but do not distinguish late producer work from network
delay. Keep first/last Put and final flag wait alongside cumulative counters.

Python build_all installs these four CLEAN versions in py311_zyq. Select one
with --variant and use the same --sync mode for all comparisons; default device
and explicit hccl remain separate controlled experiments. The new prefetch
variant has host-model coverage only until CANN/NPU validation on target hosts.

CPU checks execute the production adapters under delayed DMA and NIC reads,
including partial tails, queue wrap/capacity fallback, 4/16 ranks per host,
two/four hosts and stage recycling. SDK stubs validate call/order/address
contracts, not actual NIC encoding or CANN compiler behavior. CANN build,
device correctness and latency still require the target machines.

## 2026-11-05：BS32 输入预取实验（待 NPU 验证）

EP32/HCCL 最新干净版实测 mean：BS32 的 wqe-prep=58.16 μs，
wqe-prefetch=58.28 μs；后者 BS48/64/72 为 66.23/74.04/78.18 μs。
因此保留 base 和这两个对照，新增两种隔离实验，目标是缩短 M4
贡献加载的依赖路径；不把旧设备同步的 profile 时间直接套到 HCCL 测量。

- **input6 / ready=12**：基于 wqe-prep，M4 输入槽上限 3→6。
  H3584 每槽 7200 B，共多用 21600 B；240 KiB 接收 UB 上限不变，
  动态布局仍校验空间，较大 H/K 时允许减少槽数。目标 H3584/K8
  保留 8 个输出槽；M5 按新布局计算复用与退休 mask。
- **input-ready / ready=13**：在全部任务暂不可执行时，Vector 选一个
  未完成任务（跨 server 优先），已 ready 的成员发起 MTE2 读取。
  最多预取首 6 项，只有 resident metadata cache 可用时启用。
  某项任务变为可执行后，若它就是候选则复用读；若不是则通过
  MTE2_V→V_MTE2 释放候选槽，优先执行当前就绪任务，不阻塞进度。
  累加仍按原成员序，未改变权重、FP32 中间计算、BF16 cast 或最终完成语义。
- 不增加包、WQE 或 ready 通知，不改变 SQ/QP 所有权、GM 暂存复用和末尾 quiet。
  没有引入“读到旧数据再重试”的推测读取，跨机数据仍沿原路径回传。
- 接收输入使用 event 0–5；event 7 仍用于控制/元数据。候选最多一个，
  读取使用固定环形槽，未消费前不可覆盖。缓存容量不足时退回普通按任务调度。
  支持 Group2/4/16 和四机 EP64，不假设只有一个远端 server。

默认五个干净版本为 base,wqe-prep,wqe-prefetch,input6,input-ready，
另带对应五个 C++ 打点版本；ready-snapshot 等旧实验保留可选。
Python build_all 仅编译安装五个干净版本，run_all 共用一次初始化，
使用 --sync hccl 与当前结果对照。

schema18 的 input=lanes/early_reads/used_reads 可量化提前读是否被复用，
issued-used 表示被安全丢弃的读。还要联合观察 partial_get_wait_ticks、
partial_end、rdma_first_put/last_put 与干净 Event mean/P50/min/max。
若提前读几乎未触发或取消过多，不能仅凭输入槽增加判断有效。
这是待测假设，不能保证减少 5 μs。
