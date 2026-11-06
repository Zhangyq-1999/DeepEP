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

## 2026-11-05：EP32 HCCL 结果与 M4 实验

采集版本 c252b8e；run `ep32-py-20261105-183859-ca8c3686`。
EP32/H3584/K8/E384/BF16/random seed0/capacity256，HCCL gate，10 warmup、
100 clean samples。每轮取全 rank 最大值；无样本剔除。全部精度 PASS/error=0。
下表为 **mean / P50 / min..max，μs**。此次没有对应细粒度打点，
不能把旧 device-sync 的阶段数值当作本次 HCCL 的阶段耗时。

|BS|base|wqe-prefetch|input6|input-ready|
|---:|---|---|---|---|
|32|60.28 / 59.55 / 56.47..70.45|58.61 / 58.01 / 55.11..68.39|57.67 / 57.11 / 53.45..67.84|59.09 / 58.79 / 56.51..72.41|
|48|68.91 / 68.66 / 65.96..78.52|67.79 / 66.46 / 62.33..84.71|66.35 / 65.96 / 61.91..77.45|67.32 / 66.89 / 63.05..76.78|
|64|78.81 / 78.56 / 75.49..92.45|75.54 / 74.74 / 70.96..85.13|73.79 / 73.46 / 68.53..83.40|76.20 / 75.56 / 72.23..93.85|
|72|83.00 / 82.72 / 79.07..93.04|78.20 / 77.72 / 73.20..89.67|78.16 / 78.11 / 74.13..87.48|79.71 / 79.29 / 75.87..90.24|
|256|195.85 / 195.58 / 190.24..203.11|181.08 / 181.33 / 177.59..184.96|183.37 / 183.05 / 178.89..191.19|183.58 / 183.33 / 179.33..190.88|

`wqe-prep` 同轮 mean 为 58.65/66.92/75.71/79.22/187.80 μs，
均慢于 input6。input6 的 BS32 后50轮 P50=56.95，接近全100轮57.11；
丢弃前半段不能解释或实现所需的另5 μs改善。

本轮保留 base、wqe-prefetch、input6，新增两个隔离实验；旧版本仍可选：

- **input-pair / ready=14**：与 input6 相同的六输入槽和三块 FP32
  工作区。相邻两项分别 Cast 到 cast_/product_，一次屏障后发出两个
  独立 Muls，再按原序 Add。Muls 只作同址完整覆盖，不作部分重叠；
  符合 [Ascend Muls 地址重叠约束](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0055.html)。
  n 项贡献的显式 PIPE_V 从 3n 降为 3n−2⌊n/2⌋，例如8项24→16，
  这不是硬件执行时间或加速比例。原成员顺序、FP32乘法/加法、最终
  BF16/FP16 Cast 均保留，不使用 FMA 或树状规约。奇数尾项单独处理。
- **input-pipe / ready=15**：与 input6 相同的六输入槽；沿用 ready11
  提交流水。只有下一条任务已就绪且输出槽多于一条时暂存待发包，
  下一条的 MTE2 发起后等上一包 MTE3，然后发布 WQE。末任务、单输出
  槽立即提交；M5 前排空待提交包；全128KiB GM stage及尾部 quiet 保留。
  此版本不包含 input-pair 的计算变化，便于单独归因。

两者均不改变 Notify/Dispatch/SHMEM、通信字节、包格式、QP 所有权和
M5 规约。不增加 UB，相对 input6 的布局保持一致；覆盖 Group2/4/16、
2/4主机，包含 EP64，不把远端 server 写死为唯一一个。

schema19 的记录为272 words，旧字段位置不变。新增
`partial_pair_groups` 和 `partial_vector_barriers`，分别计数 M4
两项配对及显式 PIPE_V（含最终 Cast 后屏障），以 `vec=groups/barriers`
紧凑回显并写入 CSV/Perfetto。它们是指令调用计数，不能作时间相加。
结合 `partial_compute_issue_ticks`、local/get wait、M4 end、first/last Put、
最终flag拖尾及干净 Event 判断；不因为屏障减少就预先宣称性能提升。

默认仍为五个干净版及五个对应 C++ 打点版，Python 仅编译五个干净版，
同一进程初始化一次并顺序测试。两机都需重编，避免混用旧profile stride。
CPU 测试检查实际 Aggregate 的 BF16/FP16 逐位结果、奇数尾、1–6输入槽、
环形复用、Vector RAW/WAR/WAW屏障，以及延迟MTE/NIC与M5复用生命周期。
设备编译、正确性和时延仍需目标 CANN/NPU 实测。

## 2026-11-05：9b3d243 结果与首项计算实验

run `ep32-py-20261105-190818-14e57f59`；EP32/H3584/K8/E384/BF16，
random seed0/capacity256，HCCL gate，warmup10、100 clean samples。
逐轮 global rank-MAX，未剔除样本；全部 PASS/error=0。
下表是 **mean / P50 / min..max，μs**。

|BS|base|input6|input-pair|input-pipe|
|---:|---|---|---|---|
|32|59.21 / 58.86 / 56.91..70.96|57.28 / 56.80 / 54.25..70.85|58.69 / 56.96 / 53.57..81.26|57.60 / 56.83 / 53.99..72.05|
|48|69.10 / 68.77 / 66.37..78.92|65.49 / 65.28 / 62.09..73.45|66.18 / 65.93 / 62.19..76.76|66.51 / 65.57 / 61.44..81.35|
|64|79.27 / 78.64 / 75.33..97.56|74.26 / 73.57 / 70.08..87.65|74.69 / 74.11 / 71.05..89.71|74.17 / 73.59 / 70.70..88.69|
|72|83.19 / 82.64 / 79.16..99.24|77.83 / 77.01 / 73.98..88.66|78.43 / 77.59 / 74.05..91.93|77.21 / 76.83 / 71.49..87.94|
|256|195.42 / 195.22 / 190.96..200.48|183.79 / 183.59 / 178.75..192.05|183.78 / 183.36 / 177.85..190.95|174.94 / 174.63 / 170.06..179.92|

同轮 wqe-prefetch 的 mean 为59.10/67.06/75.02/79.52/181.19 μs。
input-pair 的小 BS 均值没有改善；input-pipe 的主要收益在 BS256。
BS64 的 rank0 P50 从 input6 的68.14降到 input-pipe 的64.80，但全rank
P50 73.57→73.59，说明局部收益没有缩短全局关键路径。此轮无细粒度
打点，不能据此定论 MTE2 或网络耗时。代码中的候选依赖是 input-pair
先等第二项输入，才对第一项乘权重。下一轮拆开实验：

- **input-stream / ready16**：继承 input6 的六输入槽与立即回传流程。
  Cast 首项后先发起 Muls(sum)，后续贡献交替 Cast 到 product/cast。
  当前 Cast 和上一项 Muls/Add 的目标互不冲突，共用随后的 PIPE_V。
  然后原位 Muls、屏障、按原序 Add；最后显式屏障保护最终低精度 Cast。
  n项显式 PIPE_V 为2n+1，原版 input6 为3n，8项24→17。
  这是发射屏障计数，不等于加速比例；每项乘法、加法和最终取整保持原序。
- **input-front / ready17**：保留 input6 的算术与屏障。
  首次只发起本卡anchor的 MTE2，Cast 后先发 Muls(sum)，再补发贡献1..
  min(n−1,inputLanes)。当n大于槽数时，最后一项补发回填已释放的首槽。
  后续按原环形流水继续，每项仅读一次。可能减少首项发射前的Scalar
  下发开销，也可能因其他输入发起较迟而退化，必须由设备实测判断。

两项均在任务所有成员ready后才读数据；保留六输入槽、八输出槽及
三块FP32工作区，UB与input6完全相同。V_MTE2保护输入复用，MTE3事件
跨M4/M5保留；全128KiB GM暂存和末尾CQ回收不变。无新notify、WQE包、
QP、跨机同步；支持Group2/4/16与多server，不以EP32或唯一远端写死。
冻结base、input6及input-pipe的执行路径保持不变。

默认五个干净版为base,input6,input-stream,input-front,input-pipe；
C++另带五个对应打点版，Python仍为五个干净版并共用一次初始化。
schema20追加partial_stream_steps和partial_front_loads，记录仍为272words。
回显arith=steps/loads、CSV及Perfetto均提供新计数。旧CSV可读，
两机仍需重新编译所选版本。HCCL同步方式保持一致后再比较干净时延，
不能将不同同步方式的C++打点时间直接解释成此次Python的阶段耗时。

CPU检查执行实际Aggregate的FP16/BF16顺序规约、1–6输入槽环形复用、
Vector RAW/WAR/WAW检查，并断言首项Muls发生在第二项输入等待之前。
front版还检查其他输入在首项Muls后发起。异步DMA/NIC测试验证源数据
生命周期，调度覆盖四机EP64。没有本地CANN/NPU，实际编译与收益待测。


## 2026-11-06：input-interleave（ready18）

本轮默认覆盖前一节的 5+5 列表，改为 base/input6/input-interleave/input-l2 与对应
C++ 打点版，Python 为四个干净版。保留 input6 的六输入槽、八输出槽、
快照建表、就绪筛选、预构造 WQE、原算术顺序和 CQ 回收。
唯一算子策略差异为物理 AIV 放置：logical = physical/2 +
(physical 为奇数 ? 0 : active/2)。原逻辑半区和 notify 索引不变。

56 AIV 的发送组是 1,3,...,55，接收组是 0,2,...,54；通知任务由
40,42,...,54 执行。M2 仍由全体核准备，M4 的规约后跨机回传仍属于
接收组。SDK QP、每核 GM 源暂存、magic 和 profile 原始地址全部保持
物理 GetBlockIdx，防止错用逻辑核导致其他核提前复用源数据。拖尾
producer map 和缺失准备标记诊断也转换为物理核号。

ready18 仅允许偶数核等分，不等分 receive 配置会拒绝执行。无新的
两机/唯一远端假设，覆盖 Group2/4/16 和每机4/16 rank 的多server索引。
CPU 检查验证任务唯一覆盖、通知覆盖、正逆核映射、物理 QP 暂存隔离、
FP16/BF16 累加序、异步DMA/NIC源生命周期和CSV/Perfetto角色。
此环境没有 CANN/NPU；OS/编译器检查不能证明 OSTD 改善或实际提速。

### L2 与最新 Dispatch 参考审计

2026-11-06 读取 liyang1384/DeepEP 的 kernel/ll-ndr-ab-per-core-qp，
当前提交 cde48af。该分支 33cc038 将接收任务按行跨 source 交错分配；
这属于任务均衡，与本轮物理 AIV 奇偶放置是两个独立维度。
5aff591/f6699a3 保留 count/cumsum 默认 L2 读视图；5ffcd2a 的 payload
默认 L2 读实验已由 f00fa1c 回退，不能当成已采纳方案照搬。

原有 Combine 版本没有在构建/启动代码发现全局关闭 L2 的设置。L2_CACHE_HINT
是编译定义，不等于全局禁用。通信 stage、ready控制、同机peer读取及
回包 payload/flag 通过 SetL2CacheHint(CACHE_MODE_DISABLE) 使用旁路；
expertOut、路由表、权重和最终output未在这些位置禁用。Dispatch 的
count/scoreboard 写者与 Combine ready 的写者、清零路径不同，不能仅
删除读侧cache hint就声称修复。以下将cache策略独立成A/B，
不混入input-interleave。以上为源码配置，不是用户机器硬件cache状态的测量。

Combine 可复用 Dispatch 表，减少 count/route metadata，但仍有 M2 准备、
ready依赖、M4跨卡取数与加权规约、回传及M5最终规约。当前 M4 也会从
assist表生成本核任务计划，因此不是完全零metadata。分层减少跨机字节
却增加依赖阶段，Dispatch 的 metadata 又可能被payload传输掩盖。
两者总时延相差约10us不能单独定位为meta、L2或RDMA耗时；须同shape、
路由、dtype、entry sync和计时范围比较，并沿最慢rank的关联包路径查因。


### 同轮追加 input-l2（ready19）

用户要求与该 Dispatch 分支保留的 L2 策略一致。input-l2 继承
input-interleave，只将控制区改为默认 L2；保持原有发布、观察、清零的
同步顺序。CopyLocalPendingTokens 的 prepare 发布、NotifyServerCard
的 prepare 轮询/清零/ready发布、CombineGroupReady 的读取/退休清零
全部保持默认视图。不能只改读侧而遗漏通知通过 LlTransport::Records
强制旁路；此版用相同32B记录与512B目的步长的 MTE 写默认控制视图，
并检查目标属于本物理host。共享epoch/ACK辅助Tensor也遵循同一策略，
但ready19实际仍使用原private-ready协议，不新增ACK等待。

payload与data-with-flag不跟随开启：M2 stage写、peer读取、RDMA暂存、
M5回包数据及块尾flag仍旁路；原先默认缓存的expertOut/路由/权重/output
保持。对照参考仓5aff591与f6699a3的控制策略，不迁入已回退的5ffcd2a。
改变控制视图的cache一致性/可见延迟必须在目标NPU验证，CPU模型不模拟
硬件coherence。正式性能比较使用相同输入/同步/计时范围和干净版。

ready18与ready19分别用于分核、控制区L2的单变量对照，冻结base及input6
保持，所有开关编译期固定。构建身份扩展到ready19，不占poll-nop位。
summary.csv写core_layout/control_l2，Python ID也输出，原诊断和Perfetto
继续提供physical AIV核号、notify/group wait与RDMA尾部时间。

## 2026-11-06：参考 zzh 分支，保留 input6 的独立对照

审计目标为 [Ziha0-Zeng/DeepEP 的 fb97fbf](https://github.com/Ziha0-Zeng/DeepEP/commit/fb97fbfa42cbae17e67d5d27ebc68d6e4b4b18e5)。
该提交的主要 kernel 修改是将 remote-first 从 Group16 扩展到物理
每机4 rank 的跨机 Group4；其余大部分是多版本共享初始化和测试入口。
构建脚本还提供 recv36、small-batch、early-reduce、remote-first 和
deferred-return。与当前分支逐项核对：

- remote-first 的本机后置、远端相对顺序保持，我们已有按物理 host
  分组的实现；每机4/16 rank 均适用，不需要迁回运行时 crossHostGroup
  参数。参考提交没有新增多 SGE 或新的 L2 策略。
- input6 已有快照建表、Vector 就绪任务筛选、六输入槽与 WQE 预构造；
  参考分支没有这些更新。不能整文件替换，否则会退回较早的规约流水。
- input6 已在已就绪任务集合中先选跨机任务；本轮 remote-first 是将
  本机后置进一步应用于 DIRECT 遍历和 M4 建表/轮转分配，也会改变
  任务所属核。不要把它描述成首次实现“跨机任务优先”。
- 延期回传在本仓 return/input-pipe 中已有对应实验。input-pipe 曾在
  BS256 有明显收益，但 BS32–64 尚无稳定收益，故保留选项、不默认叠加。
  参考 r36 的方向用于重新检验接收核配比，选择32而非36是为了对齐
  BS32/64 的 M5 token 分片，并保留更多 DIRECT 核。
- HCCL 首次 all_reduce 完成后再初始化 SHMEM、全版本共用一次初始化
  均已在当前 Python 用例实现。本轮不改同步、计时口径或初始化顺序。

最近两个 run 均为 H3584/K8/E384/BF16/random seed0/capacity256，
HCCL gate、warmup10、100 clean samples、逐轮 global rank-MAX Event。
全部精度检查 PASS；未删除任何样本。以下为 mean，单位 μs。

|规模 / BS|base|input6|input-interleave|input-l2（奇偶＋L2）|
|---|---:|---:|---:|---:|
|EP8 / 32|51.19|49.25|49.36|49.51|
|EP8 / 48|60.63|57.20|57.89|58.05|
|EP8 / 64|68.65|64.77|67.43|67.41|
|EP8 / 72|72.78|68.37|72.09|71.51|
|EP8 / 256|166.58|159.89|171.34|172.15|
|EP32 / 32|59.46|56.94|58.79|58.25|
|EP32 / 48|69.08|66.03|70.29|67.91|
|EP32 / 64|78.72|74.22|79.20|78.76|
|EP32 / 72|82.14|77.81|82.12|82.36|
|EP32 / 256|195.38|183.24|194.08|194.33|

run 分别为 `ep8-py-20261106-095739-c6bdd4ab` 和
`ep32-py-20261106-100524-22796ef8`。奇偶布局没有超过 input6；EP32
BS48 在奇偶布局上开控制 L2 有改善，但这不是原布局开 L2 的直接证据。
这些是干净 Event 数据，没有同轮细粒度阶段数据，不能据此判定 OSTD、
缓存或 RDMA 本身的耗时。新的单项实验如下：

|版本|ready|接收核（56 AIV）|布局|控制区 L2|REMOTE_FIRST|
|---|---:|---:|---|---|---|
|base|0|28|半区|旁路|OFF|
|input6|12|28|半区|旁路|OFF|
|input6-l2|20|28|半区|默认|OFF|
|input6-r32|12|32|半区|旁路|OFF|
|input6-remote|12|28|半区|旁路|ON|
|input6-fused（可选）|20|32|半区|默认|ON|

默认前五个干净版，C++ 配对应五个 profile 版（base 对应 ref-profile）。
可选 fused 将三项叠加，独立测试可用：

```bash
# 两机分别编译安装；EP32 再加 --ranks-per-host 16。
bash tests/ll_python_bench/build_all.sh --variants base,input6,input6-fused
# 第一台启动，第二台复制它打印的命令。
bash tests/ll_python_bench/run_all.sh --variants base,input6,input6-fused --sync hccl
```

base 的冻结源码无修改，input6 的分支与配置保持。L2 同时覆盖本机
prepare/ready 的发布、观察、清零，data-with-flag 和 payload 仍旁路。
三项均不增加 UB、不改权重/累加顺序、wire、QP 所有权、暂存复用或
最终 CQ 回收。ready20 的构建身份使用预留 notify=3/coalesce=1 编码，
不占 poll-nop 的高16位、不与已有可选组合冲突；不实际启用这两个协议。
每机4/16 rank 的跨机索引不依赖两机假设，覆盖四机 EP64。

判据以 BS32/48/64 为主，BS72/256 检查回退；先看相同同步口径下干净
版本的 mean/P50/min/max，再用 profile 定位：

- L2：观察 prepare/local-ready wait、ready publish、group-ready wait
  和 M4 起点是否共同提前；不以单次缓存读更快替代端到端收益。
- recv32：BS32 的 M5 每核最多1 token（原最多2），BS64 最多2（原最多3）。
  同时看 DIRECT 末尾、M4 规约/回传与 M5 最慢核；DIRECT 从28减至24核
  可能抵消收益。此前 ready4/recv32 无稳定收益，需要本轮独立复测。
- remote：看建表耗时、跨机 first/last Put、最终 flag 尾部，而非只看
  group-ready wait。任务重新分配可能降低或增加核间不均衡。
- fused：仅作为叠加实验，不能从三个单项各自快就推断组合必然快。

复用现有 schema20 的原始时刻和计数，CSV、Perfetto 保留完整记录；
累计等待/构造 ticks 不能相加当作流水块，也不能直接当作网络单包耗时。
Python ID 增加 ready/remote_first，保留 recv/layout/control_l2。
默认五个 BS、十个 C++ 版本的完整合成诊断通过小于62KiB检查，额外 BS
或版本保留完整输出，可分次粘贴。

本地验证覆盖新组合的完整 kernel Host-stub 编译（clean/profile、
Group2/4/16、每机4/16 rank）、实际建表与原扫描等价、空任务/最大BS、
四机分配、32接收核通知覆盖、UB边界、FP16/BF16累加序、延迟DMA/NIC
源生命周期、Python双模块共享初始化、冻结base和Perfetto回归。
新组合没有本地 CANN/NPU 实测；Host 模型不模拟 L2 硬件一致性，不能
保证收益。所有参与节点需重新编译同一组版本后测试。


## 2026-11-06：加入跨机优先的首末包即时回传对照

本节覆盖上节五版默认清单。新增两项都基于 input6-remote，保持六输入槽、
WQE 预构造、半区布局、28 接收核和控制区旁路；不同时叠加 L2/recv32。

|默认新增版本|ready|上一包提交点|首包/末包|
|---|---:|---|---|
|input6-rpipe|21|下一条已就绪任务的初始 MTE2 发出之后，规约之前|立即|
|input6-rfull|22|下一条任务完整规约和 Cast 完成之后，打包之前|立即|

参考分支的 defer-return 允许上一条 MTE3 暂存与下一条规约重叠，但延后首包
可能增加小 BS 首发等待。本轮用已有 CombineStreamDefer 规则保护每批已就绪
任务的首、末包；只有至少三条任务、两个以上输出槽时才延后中间包。这里的
“批”是当前已筛出的 runnable 集合，不是每个 peer 的首末包，也不是整个 kernel。
不会把未下发包带入下一次 ready 轮询或 M5 flag 等待。

与旧 input-pipe（ready15）相比，rpipe 保护首包并增加跨机优先建表/任务分配；
rfull 进一步将上一包提交推迟到下一条规约后。它比参考实现的下一次 Send 内
排空更早，在当前结果打包前就发布，避免等待当前打包与 V→MTE3 事件。
两者都可能增加中间包下发延迟，只是待验证的重叠实验，不能保证 BS32 变快。

生命周期沿用 TransferBatch：上一包 MTE3_S(7) 未退休时，不复用其输出槽及
独立 128 B WQE UB；当前任务使用独立输入槽、cast/product/sum/low。先完成
上一包 GM 写入，再发布 SQ/doorbell。GM 仍使用每核原 staging ring，满容量
退休已使用 peer 的 CQ 后才能循环覆盖，最后 CQ 在 kernel 结束前回收。
不合并 WQE、不改消息/通知协议、归约顺序、QP 所有权及 SHMEM SDK。
远端索引仍按实际每机4/16 rank 和 server 数计算，无两机配对假设。

打点复用 schema20，CSV/Perfetto 无格式升级：

- 两版均保留 deferred_packets、defer_gap、stage_wait、WQE prepare/fallback、
  first/last Put、M4/M5 和最终 flag 拖尾；不是只量 Put API 的调用成本。
- rpipe 的提交计入 early_issue_ticks / partial_prefetch_submit_calls；
  rfull 的提交计入 ready_pipeline_issue_ticks（紧凑回显 q 最后一组）。
- partial_eager_packets 计数每批首个远端任务；不把它当所有立即包数量。
  runnable 不足三条时 deferred=0 是预期行为。
- 累计等待与提交 ticks 会重叠，不能相加当作端到端 RDMA 单包时间。

默认 Python 编译并运行七个干净版；C++ 为七干净 + 七 profile。base 和 input6
保留，初始化仍只有一次。新增回显分段器按 BS 边界输出不超过62KiB的块，
超大的单 BS 按行续块，重复图例且不丢字段/样本。小于限制的原输出不变。

本地检查包括完整 clean/profile kernel 的 Host-stub 编译、构建身份唯一性、
每机4/16卡的延迟 DMA/NIC 生命周期、SQ wrap/admission 回退、GM复用、
真实 Aggregate 的输入发起/完整规约提交边界、四机64P任务覆盖、首末包/
单输出槽/空任务、FP16/BF16累加序、UB容量和 M5 前后退休。
脚本/报告检查覆盖七版本一次初始化和14阶段调度、全部 BS 诊断、分页无损恢复。
尚无 CANN/NPU 实测；需要各节点重编后按相同同步方式比较干净性能。
