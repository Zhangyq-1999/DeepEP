# LL Combine C++ 性能用例

## 用途与依赖

本用例直接调用仓库里的 Dispatch 和 Combine kernel，支持单机 EP2、EP4、EP8、EP16，以及双机 EP32。MPI 负责 SHMEM 初始化、Host 同步和结果汇总，SHMEM 负责设备通信。编译和运行都使用 C++、CMake 与 Shell，无需安装 Python 包或复制 Conda 环境。

两台机器需要相同的源码、编译选项和 SDK：

- Ascend950、CANN 9.2.0 或 9.3.0，以及 C++17 编译器和 CMake 3.20 以上版本。
- `/usr/local/mpich-3.2.1`。用例与 SHMEM 的 MPI 插件需使用这套 MPI。
- 当前分支配套的 HNS_1825 SHMEM SDK。`SHMEM_ROOT` 指向包含 `include`、`lib` 和 `deepep-sdk.json` 的安装目录，`lib` 下需有 `aclshmem_bootstrap_mpi.so`。

可以复用此前编译 LL 算子时生成的 SHMEM SDK。这个 SDK 独立于 Conda。没有现成 SDK 时，下面的 Shell/CMake 脚本会编译仓库锁定的 SHMEM 子模块，并登记 SDK 信息。

## 拉代码和编译

你已经拉取 fork 时，在现有仓库中执行以下命令。双机测试需要两台机器使用相同的绝对路径。

```bash
cd /path/to/DeepEP
git fetch origin
git switch perf/ll-combine-small-batch
git pull --ff-only origin perf/ll-combine-small-batch

# 按本机安装位置加载 CANN，例如：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MPI_HOME=/usr/local/mpich-3.2.1
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$MPI_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL=4

# 首次编译配套 SHMEM；已有完整 SDK 时校验后复用。
bash tests/ll_cpp_bench/prepare_shmem.sh
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$LD_LIBRARY_PATH"

bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --mpi-home "$MPI_HOME" \
    --build-dir "$PWD/build/ll-cpp-base"
```

`build.sh` 编译独立可执行文件和 LL kernel 库。CANN 的 Device 编译器与 Host C++ 编译器使用各自的构建目录。构建结果保存在 `build/ll-cpp-base`，其中 `bench-build.txt` 记录源码摘要和实验开关。

如果已有其他目录下的配套 SDK，可以跳过 `prepare_shmem.sh`，直接设置实际的 `SHMEM_ROOT`。脚本会保留已有构建；如果子模块中存在尚未登记的 `build/install`，它会停止并提示使用现成 SDK 或新 checkout。

base 和 opt 使用各自的 Combine kernel 源码。base 从上游提交 `5b38bf2ea1f67f38c0c2d3662618ae1a0457bfc6` 提取原始 kernel 及其依赖头文件，保存在构建目录中。原始文件保持不变，实验开关和打点只进入 opt。`bench-build.txt` 记录实际源码路径、原版提交和编译选项。

`--variant base` 明确选择原版；`--variant opt` 选择实验版。省略参数时，脚本根据实验开关选择：全部关闭时使用 base，启用任意实验或打点时使用 opt。三个优化开关可以单独编译：

| 编译参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--small-batch ON` | OFF | 缓存 FIRST_HOP 任务列表，减少重复扫表和空闲缓冲槽处理 |
| `--recv-cores 8` | 0 | 指定接收、汇集和最终归约核数；0 沿用核数均分 |
| `--early-reduce ON` | OFF | BS ≤ 72 时，按原 slot 顺序归约已到齐的贡献，同时推进至多两个 token |

显式接收核数支持 8～36，且不超过实际 AIV 核数的一半。其余核负责暂存和 DIRECT 回传。各 rank 使用相同的核数配置；最后 8 核仍负责组内就绪通知。

例如，构建一个组合实验版：

```bash
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --build-dir "$PWD/build/ll-cpp-opt" \
    --small-batch ON --recv-cores 8 --early-reduce ON
```

分别构建原版、单项优化版和组合版，再用相同参数对比。正式性能构建使用 `--profile OFF`，编译器会移除诊断代码。打点用于定位问题，启用打点后的 Event 耗时单独统计。

## 单机 EP2、EP4、EP8、EP16

每台机器执行一次 `launch.sh`。脚本启动本机所有 rank，等待任务结束并检查结果。先用两张卡跑功能冒烟；`--devices` 的编号数量与本机进程数一致。

```bash
# 上游原版只有 G4/G16 特化，EP2 功能冒烟使用 opt 的 G2 入口。
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --mpi-home "$MPI_HOME" \
    --build-dir "$PWD/build/ll-cpp-opt" --variant opt
bash tests/ll_cpp_bench/launch.sh --ep 2 --variants opt --run-id ep2-smoke -- \
    --devices 0,1 --tokens 32,48,64,72 --hidden 3584 --topk 8 --experts 384 \
    --warmup 2 --iters 5
```

EP4、EP8 同样指定 `--ep 4` 或 `--ep 8`。省略 `--devices` 时，程序默认使用从 0 开始的相应数量的卡。验证小 EP 后，用一条命令对比 16 卡原版和优化版：

```bash
bash tests/ll_cpp_bench/launch.sh \
    --ep 16 --variants base,opt --run-id ep16-ab -- \
    --tokens 32,48,64,72 \
    --hidden 3584 --topk 8 --experts 384 --capacity 256 \
    --dtype bf16 --route random --seed 0 \
    --warmup 10 --iters 100
```

`base` 使用 `build/ll-cpp-base/ll_combine_bench`，`opt` 使用 `build/ll-cpp-opt/ll_combine_bench`。脚本按指定顺序运行，前一版成功退出后再启动后一版。也可以用 `--variants base,opt,base` 做 A/B/A 对照。每次测试使用新的 `--run-id`。

EP2 使用新增的 G2 Plain16 特化，支持 BF16 和 FP16；EP4、EP8、EP16 使用 G4。单机数据走 MTE。EP32 沿用 G16，跨主机回传走 RDMA；FIRST_HOP 汇集保留 Get 方案。

小 EP 验证覆盖本机通信和归约；跨主机 RDMA 与对应中转仍需通过 EP32 验证。本次小 EP 入口在 C++ 用例中提供，原 Python 用例的 EP 范围保持不变。

## 双机 32 卡

**两台机器各执行一次 `launch.sh`，共同组成一个 32-rank MPI 作业。** 脚本封装 MPICH 3.2.1 的 Hydra 启动流程，自动启动本机代理和 16 个 rank。运行时无需 SSH，也无需复制代理命令。

两台机器先加载前文的 CANN、MPI 和 SHMEM 环境。两边使用相同的仓库绝对路径、编译结果、`--hosts`、`--master-port`、`--variants` 和 `--run-id`。下面的 `HOST0`、`HOST1` 替换成实际 IP。

第一台机器执行：

```bash
bash tests/ll_cpp_bench/launch.sh \
    --ep 32 --node-rank 0 --hosts HOST0,HOST1 \
    --master-addr HOST0 --master-port 29671 \
    --variants base,opt --run-id ep32-ab -- \
    --tokens 32,48,64,72 \
    --hidden 3584 --topk 8 --experts 384 --capacity 256 \
    --dtype bf16 --warmup 10 --iters 100
```

第二台机器执行：

```bash
bash tests/ll_cpp_bench/launch.sh \
    --ep 32 --node-rank 1 --hosts HOST0,HOST1 \
    --master-addr HOST0 --master-port 29671 \
    --variants base,opt --run-id ep32-ab
```

用例参数以第一台 `--` 后面的配置为准，MPI 将这些参数传给全部 rank。两边都启动后，脚本先运行原版，再运行优化版。第二台可以先启动，默认等待第一台最多 300 秒。

`--master-port` 是首轮测试的 MPI 控制端口，后续每版递增 1。例如，`base,opt` 使用 29671 和 29672。MPI 数据连接使用其正常端口分配；两台机器需要允许 MPI 通信。SHMEM 通过 MPI bootstrap 初始化。

`--join-timeout` 默认 300 秒，`--job-timeout` 默认 3600 秒。脚本在超时或失败时退出，并清理本次启动的进程。第一台的退出码反映 MPI 作业的整体状态，各节点同时检查本机 rank 的 PASS 文件。启动前可加 `--dry-run` 查看命令。

`run.sh` 保留为单个 MPI 作业的底层启动入口。常规单机与双机测试使用上面的 `launch.sh` 即可。

## 采样与结果

每种 BS 先运行一次 Dispatch，生成真实的路由表和专家输入。用例把 Dispatch 输出当作恒等专家的输出，使用均匀权重执行 Combine，并检查结果是否回到原输入。正式采样前后各检查一次精度，检查按 BF16 或 FP16 的舍入误差设置容差。

单机默认使用 `--entry-sync device`。用例先在计算流中排好以下任务：

```text
本卡启动门控 → SHMEM 设备 barrier → 起始 Event → Combine → 结束 Event
```

所有 rank 提交完这组任务后，Host 通过 MPI Barrier 会合，再通过独立控制流释放各卡门控。SHMEM barrier 等待所有卡到达，然后各卡执行已排好的 Combine。设备 barrier 和门控都位于计时区间外，原版 Combine 内部保持不变。预热采用相同的启动方式。

这种安排让设备 barrier 吸收 Host 放行偏差，也避免起始 Event 执行后继续等待 Host 提交 Combine。设备调度仍可能带来少量启动间隔，实际间隔以 CANN profiling 为准。

EP32 默认沿用 Host MPI Barrier，结果记录为 `entry_sync=host`。设备门控模式目前用于单机 MTE 场景。单机也可以显式指定 `--entry-sync host`，与此前测试方式对照。

`rank-N.csv` 保存每个 rank 的全部采样。rank 0 的 `summary.csv` 按同轮最慢 rank 汇总，再计算均值、最小值、最大值、p50 和 p99，单位为 μs。汇总也记录实际核数、原版标识、启动同步方式、实验开关和打点状态。双机输出分布在各自主机，完整采样需要保留两边目录。

例如，`--run-id ep32-ab --variants base,opt` 的结果保存在 `results/ll-cpp/ep32-ab/0-base` 和 `1-opt`。`node-0`、`node-1` 子目录保存各节点的启动日志。

全部采样、精度检查和 SHMEM finalize 成功后，各 rank 写出 `rank-N.PASS`。单 rank 异常会调用 `MPI_Abort`，结束本次 MPI 作业。本用例用于功能冒烟和性能对照，完整的随机权重、路由元数据与独立 CPU golden 检查仍由原 Python 精度用例覆盖。

## 分阶段打点

打点仅在 opt 构建中提供。编译时加 `--profile ON`，运行时用 `--profile 1` 记录阶段边界，或用 `--profile 2` 进一步累计热循环耗时和轮询次数。

下面只打开打点，其余优化关闭，便于先分析原流程在实验源码中的阶段开销：

```bash
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --mpi-home "$MPI_HOME" \
    --build-dir "$PWD/build/ll-cpp-opt" --variant opt \
    --small-batch OFF --recv-cores 0 --early-reduce OFF --profile ON

bash tests/ll_cpp_bench/launch.sh \
    --ep 4 --variants opt --run-id ep4-profile-1 -- \
    --devices 6,7,14,15 --tokens 32,48,64,72 \
    --hidden 3584 --topk 8 --experts 384 --dtype bf16 \
    --warmup 10 --iters 10 --profile 1
```

需要细分时，改用新的 run-id 和 `--profile 2`。分析某项优化时，保持其余配置相同，只调整对应编译开关。原版性能使用独立的 base 构建测量。

每个 rank 的 `rank-N-profile.csv` 保存每轮、每个 AIV 核的记录。主要字段如下：

| 字段 | 观察内容 |
| --- | --- |
| `stage_begin/end`、`stage_rows` | TERMINAL 本地暂存与本核任务量 |
| `notify_begin/end`、`ready_begin/end` | 本卡准备完成后的通知，以及组内就绪等待 |
| `partial_begin/end`、`plan_begin/end`、`partial_rows` | FIRST_HOP 扫表、汇集、部分归约与回传 |
| `direct_begin/prepared/end`、`direct_rows` | DIRECT 路由准备和直接回传 |
| `final_begin/init_end/end`、`final_tokens` | 最终归约的初始化、就绪检查和归约处理 |
| `drain_end`、`pipe_done`、`flush_done` | 输出回收、管线收尾和 RDMA 批次回收 |
| `probe_calls/misses/ticks/flag_blocks` | 二级打点中的轮询次数、未就绪次数、累计开销和检查块数 |
| `direct_*_ticks`、`final_*_ticks`、`contributions` | 二级打点中的扫表、打包、已有等待和实际归约贡献数 |

时间字段使用设备原始 tick。`entry_tick` 保留原始值，其余阶段时间戳均减去本核入口时间；缺失阶段留空。一级打点中的二级字段也留空。各核记录在原有收尾完成后集中写出，业务路径沿用原有同步。

先对比同一 rank、同一轮的发送核和接收核，再结合任务量判断拖尾。`probe_ticks` 包括标记加载、检查和相关等待；`final_input_wait_ticks` 只统计原有等待指令处剩余的等待。不同核的并行耗时分别分析，跨设备原始时间戳需要先确认时钟对齐，tick 换算为时间也需要确认计数器频率。

诊断记录和导出会影响时延。确认方案后，以 `--profile OFF` 重新编译，再用默认 `--profile 0` 采集最终性能。

## 启动参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tokens` | 32,48,64,72 | 每卡 token 数，可给多个 BS |
| `--hidden` | 3584 | hidden 维度，可设为 7168 |
| `--topk` / `--experts` | 8 / 384 | 每 token 专家数与全局专家数 |
| `--capacity` | 256 | 原算子的每卡最大 token 参数，影响自动去重策略；至少覆盖所有 BS |
| `--dtype` | bf16 | 支持 bf16、fp16 |
| `--route` | random | random 为无重复随机专家，hotspot 集中命中前 TopK 个专家 |
| `--seed` | 0 | 路由随机种子 |
| `--warmup` / `--iters` | 10 / 100 | 预热轮数与采样轮数 |
| `--entry-sync` | auto | 单机使用设备门控与 barrier；EP32 使用 Host MPI Barrier |
| `--profile` | 0 | 0 关闭，1 记录阶段边界，2 增加热循环打点；1/2 需要诊断构建 |
| `--timeout` | 300 | SHMEM 初始化和控制操作超时，单位为秒 |

MPICH 手动启动行为可查阅 [Hydra 3.2.1 实现](https://github.com/pmodels/mpich/blob/v3.2.1/src/pm/hydra/tools/bootstrap/external/external_common_launch.c)；端口范围和作业超时见 [mpiexec 参数说明](https://www.mpich.org/static/docs/v3.2/www1/mpiexec.html)。
