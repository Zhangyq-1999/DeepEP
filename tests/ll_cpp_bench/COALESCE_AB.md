# RDMA hidden 聚合实验

## 比较什么

本实验测试减少跨机 WQE（Work Queue Element，工作队列元素）能否降低 Combine 时延。默认使用双机 EP8，每机设备 12、13、14、15。原版 `base` 保留冻结的上游源码；四个实验版使用独立 kernel。

| 版本 | 目标选择 | 发送粒度 |
| --- | --- | --- |
| `card-all` | hidden 的最终目标卡 | 每个发送 rank 到每个目标卡，一批一个 WQE |
| `card-row` | 与 `card-all` 相同 | 一行 hidden 一个 WQE |
| `server-all` | 对端 server 的同号卡，解包后机内转发 | 每个发送 rank 到每个目标 server，一批一个 WQE |
| `server-row` | 与 `server-all` 相同 | 一行 hidden 一个 WQE |

“同号卡”按参与通信的 rank 编号计算。当前两机都选择设备 12～15，因此设备 12 发到对端设备 12，设备 13 发到对端设备 13，以此类推。聚合范围包含一个发送 rank 的全部生产核，不跨发送卡汇集。

H3584/BF16 的 hidden 为 7168 B。保留原有 512 B 块尾 flag 后，每行占 7680 B。每行还有 32 B 路由信息，每包另有 32 B 包头和 32 B 包尾。因此单行包是 **7776 B**，包含 N 行的大包是 `64 + N × 7712` B。空目标发送 64 B 完成包，接收方无需猜测本轮是否还有数据。

## 算子流程

生产核继续计算 DIRECT 或 FIRST_HOP 的回传结果。本机目标沿用原来的机内写入；跨机目标先写入本核独占的 GM 暂存区，附上最终目标卡和结果 slot。生产核写完全部结果后发布本轮完成标记，再进入原有最终归约。

实验版从 DIRECT 核中分出 4 核，负责聚合与解包。默认 56 核时，核 0～23 处理 DIRECT，核 24～27 处理聚合通信，核 28～55 处理汇集与最终归约。全部核仍参与初始本地暂存，末 8 核仍负责组内通知。

通信核用 Vector 检查本卡生产完成标记，等齐后按目标整理连续数据。整批版一次提交整个目标的数据；单行版使用相同整理过程，再按行提交。发送源缓冲保持有效，直到 quiet 确认发送完成。单行版每 16 次提交或目标尾包执行一次 quiet。

接收核检查包头、包尾的本轮 epoch，再重新加载数据并解包。card 版写入本卡原有结果 slot；server 版通过 MTE 写入同一 server 内的最终目标卡。最终归约仍检查每个 512 B 块的原有 flag，数值计算和权重处理保持原流程。这里沿用 [目标 RoCE SO 完成约定](../ll_rdma/ROCE_SO_CONTRACT.md)，不依赖不同 WQE 之间的到达顺序。

这轮实验会等待本卡整批生产完成后再发送，且新增 GM 暂存、整理与解包。因此 WQE 变少并不保证更快。先比较 `all` 与 `row` 的干净时延，再用打点判断等待和拷贝是否抵消了提交收益。

## 编译与运行

两台机器分别更新代码并编译：

```bash
git switch perf/ll-combine-small-batch
git pull --ff-only origin perf/ll-combine-small-batch
bash tests/ll_cpp_bench/build_all.sh
```

脚本默认构建 base、四个聚合版以及对应的五个打点版，共 10 个 Combine 构建。Dispatch 仍按配置共用编译产物。首次编译 SDK 和环境配置见 [用例说明](README.md)。

第一台执行：

```bash
bash tests/ll_cpp_bench/run_all.sh
```

第二台复制第一台打印的命令，传入相同的 RUN_ID：

```bash
bash tests/ll_cpp_bench/run_all.sh 第一台打印的RUN_ID
```

默认 BS 为 32、48、64、72、256。每个 BS 预热 10 轮，干净版采样 100 轮，打点版采样 10 轮。全部版本共用一次 MPI、ACL 和 SHMEM 初始化。每轮入口默认使用计时外的 MTE/RoCE 设备同步，`--entry-sync host` 可切回 MPI 对照，详见 [采样流程](README.md#采样与结果)。

只测一组时，两边编译相同选择，并让第一台用相同版本列表启动：

```bash
bash tests/ll_cpp_bench/build_all.sh --variants base,card-all,card-row,ref-profile,card-all-profile,card-row-profile
bash tests/ll_cpp_bench/run_all.sh --variants base,card-all,card-row,ref-profile,card-all-profile,card-row-profile
```

脚本仍支持覆盖机器地址、设备、BS 和采样次数。聚合版要求跨机配置；单机测试可选择 `--variants base,ref-profile`。实验缓冲位于每个共享内存 bank 的 2 GiB 偏移处，Host 在启动前检查原有工作区与聚合工作区的边界，超过容量则报错退出。

## 看哪些数据

每个 BS 开头的表格比较干净版 Event 时延。`card-all/card-row` 用来判断同一卡的数据合包效果；`server-all/server-row` 用来判断同号卡中转后的合包效果。最后与 base 比较总收益。四个版本均关闭此前的 batch、local-first、SIMT 和 Nop 实验。

新增 `A` 行显示通信核的阶段结束时间及累计开销：

- `wait`：等待本卡生产完成。
- `gather`：读取暂存行并整理连续数据。
- `put/quiet`：RDMA 提交和发送完成等待。
- `recv_wait`：等待包头与包尾；包含对端生产、发送及传输时间，不能单独解释为网络耗时。
- `unpack`：加载收到的数据，写回本卡或转发到机内其他卡。
- `rows/wqes/bytes/unpack/forward`：发送行数、WQE 数、提交字节数、解包行数和机内转发行数。

`A` 行计数属于该核。完整逐核记录仍在 `rank-N-profile.csv`，第一台汇总中保留代表性核。`T` 行中的 producer 指原始结果生产核，聚合版由另一个通信核实际发出 RDMA。核内时间仍以本核入口为起点，不用于直接计算跨卡启动偏差。

Host 测试覆盖四种组合的空包、多轮 epoch、目标映射、WQE 数量和逐字节解包。CANN 编译、NPU 精度和性能尚未验证，当前没有聚合收益结论。
