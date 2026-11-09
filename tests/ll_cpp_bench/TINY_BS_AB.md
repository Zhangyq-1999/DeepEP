# BS1–24 Combine 对照（2026-11-09）

目标是缩短极小 BS 的固定启动/准备和轮询开销。当前没有同一轮 BS1–24
干净版与细粒度测量，不把 BS32 的阶段比例外推为小 BS 已证实的瓶颈。
较优对照是 `input6-l2-route-only`（ready28/data7）；原始 `base` 继续冻结。

## 参考实现核对

参考 [ops-transformer ubx_reserve](https://gitcode.com/zzg_code/ops-transformer/tree/ubx_reserve)，
核对 commit `d1bad1c202ce628c7249bf248f1f967de74ac195` 的
`mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h`。

| 参考做法 | 对当前实现的判断 |
| --- | --- |
| ReceiveAndCombine 对没有 token 的核在初始化接收缓冲前返回 | 当前 M5 已有空核返回；进一步把 DIRECT/M4 的固定初始化按实际任务收缩 |
| PollFinalToken 保留 waitMask，确认的贡献不再检查 | 移植“记住到齐状态”的调度思想，保留 Vector 检查每个贡献的全部块 |
| 两个 payload 缓冲，读下一份时计算上一份 | 当前已有6路 M4 输入及 M5 预取；缩为2路未必有益，本轮不混入该变量 |
| 独立跨机发送核、批量 WriteNbi、尾部不足批量也提交 | 独立发送核需要新的交接；现有每核QP和完成协议不同，不能直接替换或据此认定收益 |
| scatter/gather 后全核同步，再全核最终规约 | 本仓是 M4→M5 无额外全核屏障的流水，本轮保留 |

相对此前固定参考 e6c7d2e，该 Combine 文件的最近变化主要是批量接口及尾批
CQ/SQ 回收处理（+5/-11行）。不能把该分支其他算子的“多QP”提交标题当成
Combine 已开启同样多QP的证据。现有 `ops-shmem-base` 适配器仍固定原提交，
没有静默替换它的来源或引入 HCCL 初始化/算子调用；SHMEM SDK 不改。

## 三个独立实验

| 版本 | ready | BS≤24 行为 |
| --- | --- | --- |
| input6-l2-tiny-setup | 31 | DIRECT仅初始化本核将使用的槽；M4将元数据DMA与所需输出flag初始化重叠，单任务省去依赖列表压缩 |
| input6-l2-tiny-probe | 32 | M5单token核记忆已完成贡献，只重读未完成贡献的flag |
| input6-l2-tiny | 33 | 两者组合 |

所有新版本 data_l2=7、28/28 分核、6个 M4 输入槽、原 server/核/QP 归属保持。
BS>24 使用 route-only 的原调度/初始化/轮询。即便相同执行路径，编译生成
代码仍可能有差异，因此保留 BS32–256 干净版回归测量。多 token 核或 K>8
使用原整体轮询；该条件不是 EP 限制，双机4/16卡及四机16卡都适用。

setup 复用现有 lean/single 的已测机制，但只作用于 BS≤24。DIRECT空任务核
先完成选表及其DMA退休，再返回；M4输出环的全部 MTE3_V 事件仍由调用者
初始化并传给 M5。仅省去不发送槽的 Duplicate，不改变环大小、事件清理、
GM分配或CQ退休。单任务仍使用 Vector group-ready 轮询，只把依赖判断变为
一个位掩码，不使用 SIMT 逐flag筛选。

probe 每核只保留一个 pending mask。按原槽位置合并连续的未完成贡献发起
strided DMA，每个贡献用 Vector ReduceSum 检查全部512B块，随后一次 V→S
读取至多8个和并清除已到齐bit。原 work_ 至少256B，容纳8个32B对齐结果；
M4输入/计算已经退休，M5尚未读取payload，故复用 cast_ 做向量规约临时区。
不增加UB，不降低输入lane数。Scalar读取结束后显式 S→V/S→MTE2 保护复用。

全部贡献到齐后，仍按原top-k槽顺序读取和累加；不按到达顺序改变浮点和，
不提前清flag，不跳过尾块，不以CQ代替入向完成。沿用最后一token前清完整
原flag区域及 M4输出槽→M5输入槽的退休检查。pending每次调用重新建立，
不会跨bank/轮次记住旧完成状态。

风险/收益需实测：本轮多次 ReduceSum 和状态维护可能比原一次总ReduceSum更贵，
只有减少重复读取/判断足够多才可能获益，故与setup分开编译比较。

## 打点及验收

schema25仍288 words，增加 TinyPolicy、FinalSparseProbeCalls、
FinalSparseSkippedFlags、FinalSparseConfirmedSlots。回显
`tiny=policy/probes/confirmed_slots/skipped_flags`，policy位1=setup、位2=实际启用
sparse probe。跳过量只计已确认的有效贡献，不把原mask无效槽算为收益。
原ProbeFlagBlocks记录实际读取块数，ProbeLoadTicks/ProbeCheckTicks区分读取和
向量判断开销；small/direct_meta继续显示初始化槽数、单任务次数、路由缓存。
CSV/Perfetto body参数保留全部字段；不构造虚假的独立时间片。最慢M5核还可能
没有任务，需结合FinalTokens、策略位和已有各阶段代表记录分析。

以同轮 clean 的 mean/P50/P95/min/max 判断端到端收益，profile仅解释原因，
不移除慢样本。必要时反转三个clean顺序复测。默认 BS 为
1,2,4,8,16,24,32,48,64,72,256；5个clean+5个profile，一次初始化。
回显继续每段≤62KiB，扩大BS范围后可分段传回。

本地 Host 检查执行实际生产函数：空任务、单/多任务、BS24/25边界、K8/9
回退、2/4/16逻辑组及四机任务归属、H256/3584/7168、FP16/BF16逐bit结果、
独立延迟的中间/尾flag、已确认贡献不重读、M4未完成输出DMA退休、完整flag
清理和UB边界。也检查新模式构建身份不碰撞、clean/profile编译、共享Dispatch、
Python安装/启动参数、报告分块与Perfetto字段。本机无CANN/NPU，设备编译、
四代正确性检查和性能收益均待目标机验证。

## EP32 Python 命令

两台各执行：

```bash
git pull --ff-only https://github.com/Zhangyq-1999/DeepEP.git perf/ll-combine-small-batch
bash tests/ll_python_bench/build_all.sh --ranks-per-host 16
```

第一台：

```bash
bash tests/ll_python_bench/run_all.sh --ranks-per-host 16 --sync hccl
```

第二台复制第一台打印的加入命令。默认IP、py311_zyq、采样/打点和BS列表无需
重复传参。仅测小BS可加 `--tokens 1,2,4,8,16,24`；省略 ranks-per-host 即双机
后4卡EP8。C++使用同名build_all.sh、run_all.sh（目录换为ll_cpp_bench），EP32
运行加 `--ep 32 --ranks-per-host 16`，不传Python专用 `--sync hccl`。
