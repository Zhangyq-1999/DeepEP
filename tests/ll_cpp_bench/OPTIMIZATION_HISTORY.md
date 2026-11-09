# Combine 优化收益记录

整理日期：2026-11-09。目标是回答“改过什么、实测改善了什么”，保留原始 base，以及每次实验的直接对照。这里记录的是用户回传的数据；没有把尚未测试的方案写成已有收益。

## 读表口径

- 下表单位均为 **μs**，未特别标注时为 **clean 算子的 mean**。同一行的前后值来自同一轮、相同 BS/EP/同步方式的实验。
- mean 是每轮取所有 rank 的最大时延，再对这些轮次求均值。P50、min/max 是该序列的统计量；没有删掉异常样本。
- 不把不同轮次、EP8/EP32、host/device/HCCL 同步的差值串加。某项相对原始 base 的累计收益，也不能全部归给最后加入的修改。
- “有正向观察”不等于“稳定加速”。小于约 1 μs 的变化尤其需要结合 P50、尾延迟和重复实验判断；这里没有原始逐轮样本，不能给出统计显著性结论。
- 提交链接是该策略的引入位置，不保证等于实验运行时的 checkout。没有记录运行 SHA 的旧回显，不反向猜测。

## 对 BS32–64 有价值的修改

| 修改与引入提交 | 具体改了什么 | 同轮实测收益（mean） | 适用范围与限制 |
|---|---|---|---|
| [ready-task / a5443d7](https://github.com/Zhangyq-1999/DeepEP/commit/a5443d7) | 按任务实际依赖判断就绪；已就绪任务先执行，减少被整个 group 等待挡住；优先推进可执行的远端任务。 | EP8/device，base→ready-task：BS32 **53.63→53.06**；BS48 **62.17→58.90**（−3.27）；BS64 **69.79→66.53**（−3.26）。 | 多轮小 BS 观察到收益；不是消除所有 flag 等待。证据 E01。 |
| [ready-fast / bcaf0cc](https://github.com/Zhangyq-1999/DeepEP/commit/bcaf0cc) | 缓存任务元信息和依赖，避免调度时反复筛选、重建。 | EP8/device，ready-task→ready-fast：BS32 **58.85→58.17**；BS64 **71.77→70.98**。下一轮 BS32 **53.36→52.49**。 | 小 BS 有正向观察，下一轮 BS64 反而回退 0.34。证据 E02/E03。 |
| [ready-snapshot / 18cb9a9](https://github.com/Zhangyq-1999/DeepEP/commit/18cb9a9) | 复用空闲 payload UB，保存路由快照，减少反复读取路由表；容量不足走回退路径。 | EP8/device，ready-fast→snapshot：BS64 **69.46→68.04**（−1.42），BS256 **172.63→168.94**（−3.69）。 | 该轮 BS32 **52.49→52.56**，没有增益。base→snapshot 的 BS64 总收益 4.34 包含前序修改。证据 E03。 |
| [wqe-prep / f9eec20](https://github.com/Zhangyq-1999/DeepEP/commit/f9eec20) + [收尾修复 / 339977a](https://github.com/Zhangyq-1999/DeepEP/commit/339977a) | payload 搬到 GM 时，在独立 UB 中提前构造 WQE；等待数据可见后才发布。修复空 peer 扫描开销后再评价。 | EP32/device，snapshot→prep：BS48 **78.06→76.34**（−1.72），BS64 **86.29→83.84**（−2.45）。EP32/HCCL，BS32 **58.95→58.16**（−0.79）。 | device 该轮 BS32 回退 0.35；收益受同步方式和调度影响。证据 E04/E05。 |
| [input6 / c252b8e](https://github.com/Zhangyq-1999/DeepEP/commit/c252b8e) | M4 输入槽从 3 增到 6，增加可在途的输入搬运；H3584 下多用约 21,600 B UB，保留输出槽。 | EP32/HCCL，wqe-prep→input6：BS32 **58.65→57.67**（−0.98）；BS48 **66.92→66.35**；BS64 **75.71→73.79**（−1.92）；BS72 **79.22→78.16**。 | 是后续实验的重要对照；不要把不同版本的其他流水改动也算为“多 3 个槽”的收益。证据 E06。 |
| [input6-l2 / 424e459](https://github.com/Zhangyq-1999/DeepEP/commit/424e459) | 在前半/后半分核布局上单独对照控制访问 L2 策略，避免与奇偶分核混在一起。 | EP32/HCCL，input6→input6-l2：BS32 **58.61→56.37**（−2.24）；BS64 **75.40→73.48**（−1.92）。 | 该轮 BS48 mean 变差；后续轮 BS32 也没有复现 2.24 的收益，不能称为稳定收益。证据 E07/E08。 |
| [input6-remote / 424e459](https://github.com/Zhangyq-1999/DeepEP/commit/424e459) | 改变远端/本地处理顺序及工作分配，尽早供给跨机回传。 | EP32/HCCL，input6→remote：BS32 **58.61→56.85**（−1.76）；BS48 **66.73→65.07**（−1.66）；BS64 **75.40→73.89**（−1.51）。 | BS256 略退；这是当时的调度方案，不等于最新尚未测试的 DIRECT priority 实验。证据 E07。 |
| [input6-l2-stage / 254b9d9](https://github.com/Zhangyq-1999/DeepEP/commit/254b9d9) | 分离控制、TX、RX、暂存区 L2 对照，允许回传暂存数据走缓存。 | EP32/HCCL，input6→control+stage：BS32 **56.68→55.57**（−1.11）；BS48 **65.45→64.03**；BS64 **73.70→72.67**；BS256 **183.00→176.96**。 | 上述是控制+stage 的组合收益。纯 stage 对照 control，BS32 mean **57.06→55.57**，但 P50 **55.34→55.35**，主要改善均值/尾部，不能说典型时延下降 1.49。证据 E08/E09。 |
| [input6-l2-all / 254b9d9](https://github.com/Zhangyq-1999/DeepEP/commit/254b9d9) | 将 TX/RX/暂存数据的缓存策略组合起来，检查复用收益。 | EP8/HCCL，control→all：BS32 **49.07→48.54**；BS48 **56.24→55.75**；BS64 **63.93→63.43**；BS72 **68.00→67.31**。 | 同轮各小 BS 均有约 0.5 μs 改善，但不能推广为所有访问开 L2 都更快。证据 E10。 |
| [DIRECT 路由复用 / 9baf1f0](https://github.com/Zhangyq-1999/DeepEP/commit/9baf1f0)、[独立对照 / 09b90e5](https://github.com/Zhangyq-1999/DeepEP/commit/09b90e5) | 在 UB 中复用 DIRECT 路由结果，减少重复路由读取/筛选；route-only 去掉 lean 的混杂影响。 | 最新 EP32/HCCL，all→route-only：BS32 **56.46→55.75**（−0.71，−1.26%），P50 **56.09→55.32**；BS64 **72.87→72.49**。 | BS72 **76.15→77.23**，有回退。此前 route 相对 lean 的 BS32 收益 1.23，基准不同，不能与 0.71 相加。证据 E11/E12。 |
| [route-plan / 09b90e5](https://github.com/Zhangyq-1999/DeepEP/commit/09b90e5) | 小路由表 DMA 与筛选重叠，再 Gather 所选行，缩短建表串行部分。 | 最新 EP32/HCCL，route-only→plan：BS48 **64.41→63.10**（−1.31）；BS64 **72.49→71.98**（−0.51）。 | BS32 **55.75→56.43**，不适合把它直接替换成 BS32 的优选版本。证据 E12。 |

最新 E12 中，BS32 原始 base 为 **60.74**，route-only 为 **55.75**：同轮累计改善 **4.99 μs / 8.22%**。其中最近一次路由复用的独立改善是 **0.71 μs**，其余来自此前保留下来的组合优化。

## 主要改善大 BS 的修改

这些也有正向结果，但不能用它们替代 BS32–64 的优化结论。

| 修改 | 做了什么 | 同轮正向结果 | 小 BS 限制 |
|---|---|---|---|
| [stage-stream / 0504d90](https://github.com/Zhangyq-1999/DeepEP/commit/0504d90) | 首包尽早提交，后续包准备与发送重叠。 | 两轮 EP8/device，BS256：**170.26→167.11**、**168.89→165.69**。 | 小 BS 有好有坏；见逐轮明细。 |
| [wqe-prefetch / 78fdafd](https://github.com/Zhangyq-1999/DeepEP/commit/78fdafd) | 提交上一包时，提前搬入下一个已就绪任务的输入。 | EP32/HCCL，BS256 wqe-prep→prefetch：**187.80→181.08**（−6.72）。 | 该轮 BS32 仅约 0.04 的变化，不能算有效优化。 |
| [input-pipe / 9b3d243](https://github.com/Zhangyq-1999/DeepEP/commit/9b3d243) | 六输入槽与 WQE 发布流水组合。 | EP32/HCCL，BS256 input6→pipe：**183.79→174.94**（−8.85）。 | BS32、48 回退。 |
| [input6-r32 / 424e459](https://github.com/Zhangyq-1999/DeepEP/commit/424e459) | 收侧核数从 28 增至 32，重新平衡供给与规约。 | EP32/HCCL，BS64 **75.40→73.24**；BS256 **183.11→174.71**。 | BS32 回退约 0.51。 |
| [input6-l2-pipe / a051d1c](https://github.com/Zhangyq-1999/DeepEP/commit/a051d1c) | 小批任务首包与后续回传暂存重叠。 | EP32/HCCL，BS256 all→pipe：**178.74→171.74**（−7.00）。 | 同轮 BS32/48/64/72 都回退。 |

## 修复与测量流程收益，单独记录

- **收尾扫描修复 339977a**：跳过空 peer 和空 mask，保留活动 peer 的完成等待。profile 中 BS32/48/64 的额外收尾从 **2.796/2.748/2.808** 降到 **0.593/0.618/0.698 μs**，减少约 2.1–2.2 μs。它是不同轮次的局部打点对照，不能直接宣称整个 clean 算子减少同样时长。详见 [TRANSFER_AB.md](TRANSFER_AB.md)。
- **一次初始化跑多个版本/BS**：C++ 的 [aea1191](https://github.com/Zhangyq-1999/DeepEP/commit/aea1191) 和 Python 的 [bc2156b](https://github.com/Zhangyq-1999/DeepEP/commit/bc2156b) 复用通信会话，减少测试启动耗时；不是 Combine 内核加速。
- **先初始化 HCCL，再初始化 SHMEM**：bc2156b 修复 Python 流程初始化冲突。HCCL 算子前同步还改变启动偏差；不能把旧 host 同步与新 HCCL 同步之间的时延差归到 Combine 代码。
- **clean/profile 分离、Perfetto、精简回显**：改善测量可信度和定位效率，不记作算子性能收益。

## 未提升或证据较弱的方案

逐轮明细也保留了早期 remote/pipeline/return、local-first、poll800、SIMT 通知、stage-prefetch、分核比例等出现过正向数值的对照，包含同轮退化的 BS。早期 host 同步下均值改善有时主要来自长尾变化，不能当作稳定的内核收益。

多 hidden 聚合发送、共享 ready、奇偶核布局、部分 SIMT/额外投机筛选等，没有得到足够证据作为整体更优默认。降低 WQE 数量也不等于端到端时延下降。早期只有 profile 的核数实验不与后来的 clean 表混算。

最新 **51c99f8** 的 remote DIRECT 优先、TX staging bypass 尚未收到实测结果，不纳入正收益清单。

## 原始证据和完整记录

- [逐轮明细](OPTIMIZATION_HISTORY_DETAILS.md)：63 组出现过 mean 或 P50 正向观察的实验对照；每组保留其他 BS 的退化项，并列出 mean、P50、min/max、附件 ID、原文行号、实施提交。
- [可复算数据](OPTIMIZATION_HISTORY_DATA.json)：37 份回显附件索引及 SHA256，677 行 clean 原始表项，357 条同轮对照。其中 171 条的 mean 数值下降，**不是 171 项已验证优化**。有的附件只有失败/启动信息，未产生可用 clean 行。profile 行不混入 clean 统计。
- 对照差值：`saved_us = before.mean - after.mean`；百分比除以 `before.mean`。原始完整行保存在 JSON 中，P95/P99 按各回显真实字段保留，不互相替代。
- 本记录覆盖此次能够配对的历史回显；没有对照或无法确定口径的内容，不补造收益数字。文件名中的来源 ID 对应用户提供的附件。

| 证据 | 附件 ID |
|---|---|
| E01 | `e0edccc1-c96e-49d3-a96a-84c1022c01cf` |
| E02 | `8d8cdea7-0f82-434c-9694-68ff510fd0d0` |
| E03 | `f085a6cd-6cd0-44f3-8f7b-ddab757a336d` |
| E04 | `3157520b-0c24-4237-808b-9cbc4d7d2717` |
| E05 | `85b982f6-9895-4f15-9211-09c4d62baa64` |
| E06 | `004b63ce-4ab5-475c-9252-696f3ce5f1dd` |
| E07 | `b972f8d6-d47f-4797-b465-fc80d756f181` |
| E08 | `1d21bbad-d8b7-4fe6-9237-c1a025d3324f` |
| E09 | `6da65d96-a1ae-449e-a49c-001716294c64` |
| E10 | `c2448b56-dd6d-4433-955c-528dd3c413e1` |
| E11 | `fc19ee5d-2c34-4344-b997-b6fbbe91a4e6` |
| E12 | `b90b69ff-946f-4fc6-8b31-2a8755da923c` |
