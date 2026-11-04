# Python Combine 性能测试

## 测试内容

这套脚本用 PyTorch 启动多机测试，比较算子前同步对 Combine 时延的影响。默认使用 `py311_zyq` 环境，两台机器各用设备 `12,13,14,15`，总共 EP8。

| 参数 | 默认值 |
| --- | --- |
| 主机 | `90.90.93.84,90.90.93.184` |
| 算子 | 干净的 `base` |
| BS | `32,48,64,72,256` |
| Hidden / TopK / Experts | `3584 / 8 / 384` |
| 数据和路由 | BF16、random、seed 0 |
| Capacity | 256 |
| 预热 / 正式采样 | 10 / 100 次 |
| 算子前同步 | HCCL `all_reduce`，计时不包含同步 |

Python 流程复用 C++ 用例的输入、路由、精度检查和原始 Combine kernel，只新增测试扩展 `_ll_py_combine`。Dispatch 只负责准备 Combine 的输入，放在计时外。每次运行只初始化一次 SHMEM，依次完成所有 BS。

## 两台机器分别编译安装

在现有仓库目录执行：

```bash
git fetch origin
git switch perf/ll-combine-small-batch
git pull --ff-only origin perf/ll-combine-small-batch

bash tests/ll_python_bench/build.sh
```

脚本自动激活 `py311_zyq`，用该环境的 Python 编译并安装测试扩展。扩展、对应 kernel 库和构建记录安装在 `py311_zyq/lib/python3.11/site-packages/deepep_ll_bench/base/`。运行脚本自动切换到同一环境，并加载这里的安装产物。

默认只编译 `base`，并复用已有 Dispatch 编译产物。脚本保留环境中的 Torch 和 torch-npu 版本，只在缺少构建工具时安装 pybind11 或 CMake。测试扩展独立安装，环境中原有的 `deep_ep` 包保持不变。

编译和运行都会打印 `ENV=py311_zyq PYTHON=...`，安装完成后打印 `INSTALLED=...`。即使当前终端激活了 `base` 或其他环境，脚本也会切换到 `py311_zyq`；激活失败则退出。若 Conda 不在常见路径，可先设置 `CONDA_EXE=/实际路径/bin/conda`。

脚本沿用当前 C++ 构建配置，因此编译仍需要 `/usr/local/mpich-3.2.1`。Python 运行使用 `torchrun`，无需 `mpiexec`。CANN 和 SHMEM 默认路径与 C++ 脚本一致，也可以指定：

```bash
bash tests/ll_python_bench/build.sh \
    --conda-env py311_zyq \
    --cann-home "$ASCEND_HOME_PATH" \
    --shmem-root "$SHMEM_ROOT" \
    --mpi-home /usr/local/mpich-3.2.1
```

运行脚本也接受这几个环境参数。两台机器分别选择本机的 SDK 路径。

## 双机启动

第一台 `90.90.93.84` 执行：

```bash
bash tests/ll_python_bench/run_all.sh
```

脚本生成新的 RUN_ID，并打印第二台的完整启动命令。在 `90.90.93.184` 粘贴该命令：

```bash
bash tests/ll_python_bench/run_all.sh <第一台打印的RUN_ID>
```

脚本根据本机 IP 判断节点编号。默认映射为：第一台的全局 rank 0～3 对应设备 12～15，第二台的全局 rank 4～7 对应设备 12～15。每次从第一台启动都会生成新目录，连续运行无需清理旧结果。

建议先对同一份 kernel 做同步方式对照。第一台执行以下命令，再在第二台粘贴它打印的命令：

```bash
bash tests/ll_python_bench/run_all.sh --sync both
```

每个 BS 先测 `host`，再测 `device`，两种模式分别预热。`host` 使用 Gloo 的 Host barrier；`device` 使用下面的 HCCL 同步。这个对照保留相同输入和 kernel，但 `host` 模式与原 C++ 用例的 MPI 实现并不相同。

## 同步和计时顺序

同步方案参考 [Ziha0-Zeng/DeepEP 的提交 b1ed463](https://github.com/Ziha0-Zeng/DeepEP/commit/b1ed463)：Python 使用 `HcclBenchmarkGate` 的调用顺序，native 使用本地 Notify 控制任务开始执行的时机。

```text
准备输入、路由和输出（计时外）
    ↓
本地 Notify 暂停目标流
    ↓
提交 HCCL all_reduce → Work.wait → 排空 torch_npu Host 提交队列
    ↓
提交 start Event → Combine kernel → end Event
    ↓
独立控制流释放 Notify
    ↓
HCCL 完成 → start Event → Combine → end Event
```

Notify 保证本机先提交完整的计时任务，HCCL 负责跨 rank 同步。这里使用 ACL Notify 和 HCCL，没有调用先前的 RDMA 同步 kernel。HCCL 完成时间仍可能存在偏差，需要实测判断改善幅度。

启动脚本将 `TORCH_HCCL_BLOCKING_WAIT` 和兼容名称 `HCCL_BLOCKING_WAIT` 设为 `0`。这样 `Work.wait()` 只提交设备流依赖，Host 可以继续提交 Event、kernel 和 Notify 释放操作。[阻塞等待说明](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/collective_communication/TORCH_HCCL_BLOCKING_WAIT.md)解释了两种模式的区别。

可用以下短测检查人为延迟是否进入计时：

```bash
bash tests/ll_python_bench/run_all.sh \
    --tokens 32 --warmup 2 --iters 5 --gate-delay-ms 10
```

该参数只在 rank 0 提交 HCCL 前增加 10 ms 的 Host 等待。正常情况下，这段等待应留在 Event 计时区间外。此项用于验证计时顺序，不作为正式性能数据。

## 参数和其他实验

默认先比较原版的同步方式。要测聚合实验，两台机器都编译同一版本，然后在第一台启动，第二台粘贴输出的命令：

```bash
bash tests/ll_python_bench/build.sh --variant card-all
bash tests/ll_python_bench/run_all.sh --variant card-all
```

可选干净版本为 `base`、`card-all`、`card-row`、`server-all` 和 `server-row`。一次选择一个版本，所有 BS 共用一次初始化。

IP、端口、设备和 case 参数都可修改：

```bash
bash tests/ll_python_bench/run_all.sh \
    --hosts 90.90.93.84,90.90.93.184 --node-rank 0 \
    --master-port 30011 --devices 12,13,14,15 \
    --tokens 32,48,64,72,256 --warmup 10 --iters 100
```

PyTorch 默认使用端口 30011，SHMEM 使用下一端口 30012。`--shmem-ip-port tcp://IP:PORT` 可单独指定 SHMEM 入口。多网卡环境可用 `--ifname eth0` 同时指定 Gloo 和 HCCL 的网络接口。

扩展到每机 16 卡时，两台机器先用 `--ranks-per-host 16` 重编。运行时也传 `--ranks-per-host 16 --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`。`--hosts` 支持 1、2 或 4 台机器；跨机参数需保持一致。

## 读取结果

每个 BS 完成后立即输出表格，并保存到 `results/ll-python/<RUN_ID>/`。全部结束后，第一台集中打印 `LL_PY_REPORT_BEGIN` 至 `LL_PY_REPORT_END` 的汇总块。正常跑完后，复制这一块即可分析两台机器的全局数据。

汇总先给配置和时延表，再列每个 case 的 `D` 行。表格包含均值、P50、最小值、P99、最大值、末 50 轮 P50、rank0 P50 和最大精度误差。`D` 行补充前 3／10／20 轮均值、各 rank 的 P50 范围，以及最慢 3 轮的轮次、rank 和耗时。轮次从 1 开始；并列最慢时显示编号最小的 rank。`--sync both` 还输出设备同步相对 Host 同步的 P50 降幅。

| 输出 | 内容 |
| --- | --- |
| `report.txt` | 与首台汇总块相同的文本，每完成一个 case 就更新 |
| `summary.csv` | 每个 BS、每种同步方式的时延统计 |
| `bs32-device.json` 等 | 配置、所有 rank 的原始样本和汇总 |
| `rank-N.jsonl` | 本 rank 每个 BS 的样本，完成后立即写盘 |
| `node-N/console.log` | 本机回显 |
| `rank-N.PASS` | 精度、采样和 SHMEM 释放均完成后的标记 |

主表先取同一轮所有 rank 的最大值，再计算均值和 P50，单位为微秒。`tail50_p50` 只统计末 50 轮；采样不足 50 轮时使用全部样本。`rank0_p50` 单独统计 rank 0，便于核对参考仓的输出口径。原始样本全部保留，不自动剔除异常值。

前 3／10／20 轮均值在样本不足时使用已有样本。`report.txt` 在运行中标记为 `RUNNING`，完成精度检查、采样和释放后才标记为 `PASS`。若途中报错，已完成的 case 仍保存在文件中；此时还需提供报错机器的首错回显。

这里保留 C++ 用例的随机路由和均匀权重。参考仓的 Python 用例另有均衡路由及权重设置，因此比较两仓时还需核对输入和统计字段。先用 `--sync both` 判断相同 kernel、相同输入下的同步影响。

正常运行只打印初始化阶段、每个 BS 的开始和结果，采样循环不逐轮刷屏。每 15 秒的心跳仍显示当前阶段和已完成轮数，便于区分初始化、预热、采样和释放。`--kernel-timeout` 默认 10 秒，约束一次 stream 完成等待；超时会退出 worker，由 torchrun 终止同组进程。该等待也包含尚未完成的同步任务，不能仅凭超时报错判断 Combine 卡死。

Host 检查覆盖了启动参数、采样汇总、单次初始化和入队顺序。CANN 编译及双机 NPU 测量需要在目标环境执行。
