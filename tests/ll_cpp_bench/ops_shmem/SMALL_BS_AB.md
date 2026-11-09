# 小 BS 发送完成与最终规约实验

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

## 两个独立版本

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

默认保留三个干净对照及两个实验的 clean/profile，共7个构建。已经采过的参考版、route-only profile 改为按需选择；BS仍为1,2,4,8,16,24，默认回显测试限制62 KiB。

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
