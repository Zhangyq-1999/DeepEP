# 小 BS 发送完成与最终规约实验

## 当前实验：Hcomm / SHMEM 可追溯差异

按用户约定，暂按两边厂商 RDMA 扩展库一致处理，不将未知底层差异当作性能解释。核对的源码为 ops-transformer `d1bad1c`、旧 SHMEM `4c671684`、[asc-devkit 的 Hcomm AIV 实现](https://gitcode.com/cann/asc-devkit) `e2450c6e6748c8e92ab8997c26b4c586e6441749`（`src/aicore/hcomm/impl/platform_v310/hcomm_aiv_roce.h`）。Hcomm 控制面见 [cann/hcomm](https://gitcode.com/cann/hcomm)。当前 SHMEM 已升级到 [PR1000](https://gitcode.com/cann/shmem/pull/1000) 的固定 head `3bbe2a74738805a6a716df3f2a05f8ef47265ebe`，以支持 RDMA-v2 IPv6 建链；下表按新依赖重新核对，性能需整组重测，不能把依赖升级收益算成实验收益。

本轮默认7个构建：`base`、`input6-l2-route-only`、`ops-shmem-overlap` 三个干净对照，`ops-shmem-cq`、`ops-shmem-batch` 两个新实验及各自 profile。BS仍为1/2/4/8/16/24。旧 `ops-shmem-base`、retire及所有旧profile继续可选；下方历史数据不代表新实验结果。

| 对照项 | overlap | cq | batch |
| --- | --- | --- | --- |
| 最终规约/收尾顺序 | final → quiet → 退出全核同步 | 相同 | 相同 |
| CQ读取 | SDK整条CQE搬到UB、MTE2等待 | `ld_dev`读取owner/op 64-bit，`st_dev`更新CI | 同cq |
| WQE准备 | SDK逐包建WQE | 不变 | 缓存QP/MR/键，在上游已有UB批次中写64B WQE |
| SQ/DB发布 | 每包拷贝、每包DB | 不变 | 每批1次SQ拷贝（回绕2次）、1次DB |
| CQE / fence | 1 / 0 | 1 / 0 | 1 / 1，与Hcomm默认值一致 |

因此 `cq-overlap` 隔离轻量CQ读写收益；`batch-cq` 同时包含WQE构造/发布和fence默认值对齐，不能只归因于DB次数。保留原批次策略（目标76800B、至多10包），H3584单个回传token仍是7680B，不是多SGE或跨token拼成单个payload。BS1通常只有一个WQE，没有减少DB次数的空间，重点看CQ与建WQE开销。

### 内部能力与完成保障

`roce_queue.hpp` 从SHMEM现有 `rdma_info` 获取每peer/物理核的SQ、CQ及MR，直接复用已建立的连接与内存注册。旧依赖的 HNS_1825 公开批量路径受限；PR1000 的基线已带入公开聚合提交及稀疏CQE实现。这轮仍使用内部上下文和doorbell helper，固定每条WRITE请求一个CQE、fence=1，使批次边界与Hcomm实验一致；没有引入CQE压缩，也不新增HCCL接口，不修改SHMEM源码。已锁定SDK支持PI-on-chip，继续调用其SQ doorbell实现（含低PI地址编码与旧固件回退），没有把“去软件DB”计作本轮新增收益。

批量提交前检查SQ/CQ容量，保留至少一个invalid marker槽；不足时先回收已发布请求。UB→SQ完成、next-invalid marker及SQ缓存维护完成后才敲DB；提交返回才允许复用WQE UB。QP仍由一个物理核独占，发送GM仍是每token/server的独立不可变槽，final不写这些槽。结束时只回收实际发过的peer，所有出向发送完成和退出屏障均在Event内。零包核跳过。

轻量CQ代码仅用于“一条64B WRITE WQE对应一个CQE”的路径，检查owner、QPN、opcode和WQEBB序号，保留SHMEM完整32-bit head/tail镜像，并独立维护SQ/CQ计数，不假设此前使用者产生的两者计数相等。成功CQE序号读取跟随Hcomm的 `opSrWqebb[15:0]`；新SHMEM公开poll读取另一个 `wqe_counter` 字段，这项实现差异仍需设备回显验证。没有把Hcomm的16-bit SQ游标直接覆盖SHMEM游标；错误或等待超过1秒（Ascend950）打印队列位置并trap。未引入稀疏CQE、unsignaled请求或共享QP并发写入。

这轮仍保留一个明确差异：原生Combine没有显式Drain，而本实验保留退出前quiet。先在同一完成条件下比较传输实现；不会将quiet移到Event外来缩短读数。两者是否能进一步统一完成协议，需结合实际调用方的GM复用及下一轮同步约束分析。

### 新打点与验证

schema28，每核320 words，选中的库需要全部重编。`oq batch=n/max/copies`为实际DB批次数、最大批包数、SQ拷贝次数；`build`为WQE准备累计ticks；`cq=calls/loads/entries/ticks/capacity`为CQ轮询调用数、设备load次数、回收条数、累计ticks和容量触发回收次数。既有 `op` 的put/quiet及首末Put仍保留。batch版首末Put和put ticks在真实SQ/DB提交处计时，不把UB建WQE当作已发送。

全部字段写入CSV和Perfetto事件参数；累计ticks不伪装成可独立相加的时间片，quiet不是网络单程时延。默认回显继续按62KiB检查分块。

`test_ops_queue.py` 从锁定SDK提取真实结构、WQE编码和SQ doorbell函数，编译实际实验适配层，与异步NIC模型连接。检查逐字节编码（除显式fence）、先完成SQ再敲DB、UB破坏后继续发送、SQ/CQ回绕、16/32-bit计数回绕、容量回收、三远端peer/物理核隔离、六代数据变化、错误/陈旧CQE和大BS回退。模型不能验证真实NIC或CANN编译，精度与性能由目标机测试验收。

## 依据：2026-11-09 EP32 / HCCL

运行 `ep32-py-20261109-140737-56aeb397`，EP32、每机16卡、H3584、TopK8、384专家、BF16、random seed0、capacity256。10次预热，clean100次、profile10次，每轮取全局 rank-MAX。下表单位 μs，单元格为 **mean [min–max]**，不剔除样本。

| BS | 冻结 base | ops-shmem-base | input6-l2-route-only |
| --- | --- | --- | --- |
| 1 | 41.34 [38.41–55.07] | 34.87 [32.93–46.65] | 38.91 [34.22–60.27] |
| 2 | 42.55 [39.90–53.51] | 48.49 [46.56–57.89] | 39.29 [34.47–49.61] |
| 4 | 44.68 [42.33–56.48] | 54.24 [51.51–66.71] | 41.38 [37.18–60.47] |
| 8 | 48.49 [44.12–67.66] | 65.89 [63.97–75.20] | 42.46 [38.85–58.76] |
| 16 | 50.40 [48.02–66.28] | 102.36 [99.61–120.56] | 46.07 [41.37–58.50] |
| 24 | 53.64 [50.95–65.34] | 139.28 [135.44–148.99] | 51.49 [48.64–72.98] |

参考版只在 BS1 有优势；不能据此把整套算法替换为参考版。优先校准 SHMEM 适配层放大的同步成本，再判断其 server 预规约是否值得迁移。

同一 profile 样本内的证据（以下全部是 ticks/core-entry，不能直接当作 μs）：

- BS1，ops P50 的 sample4/rank10/core55：1包7680 B，first/last Put=14654/15785，sender_end=30543，quiet=14492/1次，barrier=30650–30816，final_init/first_ready/end=33219/34609/35445。final 在发送完成和全核同步之后才准备。
- BS2，ops P50 的 sample8/rank14/core55：2包15360 B，first/last Put=11025/28551，quiet=27732/2次，sender_end=41840。两包之间已出现长间隔；适配层每批 quiet 会阻塞后续扫描与提交。
- BS4，ops P50 的 sample1/rank2/core55：4包30720 B，put累计3922，quiet累计29478/2次。BS16 的代表发送核有16包，quiet累计约52–53k/3次。

发送端 quiet 包括队列完成等待，不是网络单程时间。各 rank 的核入口没有共同时间原点，不相减推导链路时间。BS1 ops clean mean34.87、profile mean50.11 μs，打点扰动明显，因此不能从 profile 阶段直接减出预计 clean 收益。用户此前原生 Hcomm 的23.01/29.28 μs也不是此 SHMEM 适配版的同传输基线。

## 上一轮的两个调度版本

| 版本 | 行为 |
| --- | --- |
| ops-shmem-base | 固定上游 d1bad1c 的调度，每个 BatchCommit 做 quiet，之后全核同步，再最终规约 |
| ops-shmem-retire | Put 仍立即下发；BatchCommit 不等待，所有本核包提交后每个实际 peer quiet 一次；保留原最终规约前全核同步 |
| ops-shmem-overlap | retire 的发包方式；各核完成前段任务后进入原最终规约，最后处理本核 quiet，并在退出前全核同步 |

实验仅在 BS≤24、EP≤64 生效，较大形状走参考版原调度。使用相同 pre-reduce、拓扑、任务所属核、QP、payload、flag/epoch、权重和累加次序；不调 L2，不新增逐包 clock，不修改 SHMEM SDK。policy0 的源码适配仍只有原来四处替换。

overlap 不是省略收齐检查：每项最终贡献仍经过 `PollFinalToken` / `IsReceiveSlotReady`，全部贡献就绪才消费。它只是让有最终任务的核更早建表、检查入向数据，不再统一等本卡所有出向 CQ 完成。BS1 最终任务落在55核，因此不会把全部发送工作隐去；仍需要 clean Event 验证剩余收益。

## 生命周期与队列检查

- pre-reduce 每个 token/server 只写一次独立 GM 源槽，最多24次 Put/peer。发送源不复用流水 UB，也不会被最终规约改写；所有出向 quiet 在 kernel 返回前完成。
- SDK Put 本身已发布 WQE/doorbell；不依赖适配层 BatchCommit 推进。保留 SDK 的 head/tail、队列容量检查和自动回收，不自行写 SQ 或绕过 admission。
- pending peer 跨 TPipe 重置保存，在真正发送时加入列表；收尾只遍历实际 peer，零包核跳过。EP≤64 的位图及最多16个 peer 的列表覆盖当前支持拓扑。
- 上游 scatter/gather/final 自己排空本核 DMA 后重置 TPipe。收尾重新分配128 B UB scratch，不能保存已被重置/复用的旧 UB 视图去 quiet。
- final 对本 server 的归约结果及远端结果仍逐项检查完整原 flag/代次。出向 quiet 不替代入向检查；退出全核同步保留。上游 ping-pong/epoch、版本切换、四代改值正确性检查保持。
- 这只保证相对适配基线的累加顺序不变。参考算法与 DeepEP 的原有分组不同，不能宣称两者浮点结果逐 bit 相同。

主机测试编译真实适配层，覆盖每核 QP、延迟完成、空批次、三个远端 peer×24包、六代地址复用、UB scratch 破坏与重建；对生成源码核对除声明/Process 外的所有 payload/flag/reduction 方法一致，并执行生成的 Process 骨架验证 clean/profile 与大 BS 回退顺序。Python native 模型检查共用初始化、策略身份及 Perfetto；这些不能代替目标 CANN 编译、NPU通信和精度验证。

## 打点与下轮判断

schema27/304 words。保留 `oe` 的 init/scatter/first_ready/gather/first-last Put/sender/barrier/final，全量 CSV/Perfetto；追加实际生效 `policy`、`deferred`（跳过批次等待次数）、`peers`（收尾 quiet 次数）、`retire` 起止。

Perfetto 中新增 `ops deferred transport retirement`。policy2 的全核同步命名为 `ops exit all-core barrier`；不把累计 quiet/put ticks 伪造成独立流水块。

下一轮先看精度和代次检查，再看 clean mean/P50/min/max。retire 应减少 quiet 次数并缩短首末 Put 间隔；overlap 应让 final 更早开始，且全部 quiet 仍在 Event 内。若 quiet 下降而总耗时不降，继续核对 final-ready/退出屏障和首包供给，不宣称 RDMA 已变快。BS>24 的回退需按需复测。

上述为上一轮调度实验；当前默认组合见文首，仍为7个构建，BS仍为1,2,4,8,16,24。

EP32 两台均编译安装：

```bash
git pull --ff-only https://github.com/Zhangyq-1999/DeepEP.git perf/ll-combine-small-batch
bash tests/ll_python_bench/build_all.sh --ranks-per-host 16
```

第一台运行，第二台复制打印的加入命令：

```bash
bash tests/ll_python_bench/run_all.sh --ranks-per-host 16 --sync hccl
```

默认自动使用 py311_zyq；省略 ranks-per-host 即原双机后4卡 EP8。全部版本/BS共用一次初始化。实验尚无设备性能结果，不预设收益。
