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

原流程默认保留。三个实验开关可以单独编译：

| 编译参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--small-batch ON` | OFF | 缓存 FIRST_HOP 任务列表，减少重复扫表和空闲缓冲槽处理 |
| `--recv-cores 8` | 0 | 指定接收、汇集和最终归约核数；0 沿用核数均分 |
| `--early-reduce ON` | OFF | BS ≤ 72 时，按原 slot 顺序归约已到齐的贡献，同时推进至多两个 token |

显式接收核数支持 8～36，且不超过实际 AIV 核数的一半。其余核负责暂存和 DIRECT 回传。各 rank 使用相同的核数配置；最后 8 核仍负责组内就绪通知。

例如，单独测试提前归约：

```bash
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --build-dir "$PWD/build/ll-cpp-early" \
    --early-reduce ON
```

分别构建原版、单项优化版和组合版，再用相同参数对比。新路径目前通过了本地 CPU 模型检查，CANN 编译、设备精度和性能仍需在目标机器验证。

## 单机 EP2、EP4、EP8、EP16

先用两张卡跑功能冒烟。`--devices` 的编号数量与本机进程数一致，可以选择当前空闲的卡。

```bash
bash tests/ll_cpp_bench/run.sh --hosts localhost --ep 2 --launcher fork -- \
    --devices 0,1 --tokens 1,8 --hidden 3584 --topk 8 --experts 384 \
    --warmup 2 --iters 5 --output-dir "$PWD/results/cpp-2p-smoke"
```

EP4、EP8 同样指定 `--ep 4` 或 `--ep 8`。省略 `--devices` 时，程序默认使用从 0 开始的相应数量的卡。验证小 EP 后，再跑 16 卡性能：

```bash
bash tests/ll_cpp_bench/run.sh \
    --hosts localhost --ep 16 --launcher fork \
    --binary "$PWD/build/ll-cpp-base/ll_combine_bench" -- \
    --tokens 1,2,4,8,16,32,48,64,72 \
    --hidden 3584 --topk 8 --experts 384 --capacity 256 \
    --dtype bf16 --route random --seed 0 \
    --warmup 10 --iters 100 --output-dir "$PWD/results/cpp-16p-base"
```

每个 MPI rank 绑定一张卡。结果目录应使用新路径，避免覆盖已有采样。

EP2 使用新增的 G2 Plain16 特化，支持 BF16 和 FP16；EP4、EP8、EP16 使用 G4。单机数据走 MTE。EP32 沿用 G16，跨主机回传走 RDMA；FIRST_HOP 汇集保留 Get 方案。

小 EP 验证覆盖本机通信和归约；跨主机 RDMA 与对应中转仍需通过 EP32 验证。本次小 EP 入口在 C++ 用例中提供，原 Python 用例的 EP 范围保持不变。

## 双机 32 卡

**两台机器共同组成一个 32-rank MPI 作业。** MPI 模式使用自己的 bootstrap，参数中的地址和端口范围用于 MPICH，无需另设 SHMEM TCP 端口。

### SSH 可互通时

在第一台机器执行。将 `HOST0`、`HOST1` 替换成两台机器可互通的 IP，`HOST0` 同时作为启动端地址。

```bash
bash tests/ll_cpp_bench/run.sh \
    --hosts HOST0,HOST1 --master-addr HOST0 \
    --port-range 29671:29771 --launcher ssh \
    --binary "$PWD/build/ll-cpp-base/ll_combine_bench" -- \
    --tokens 1,2,4,8,16,32,48,64,72 \
    --hidden 3584 --topk 8 --experts 384 --capacity 256 \
    --dtype bf16 --warmup 10 --iters 100 \
    --output-dir "$PWD/results/cpp-32p-base"
```

脚本为每台主机启动 16 个连续 rank。两边需要相同的源码、二进制、SDK 路径，以及可用的动态库搜索路径。

### 两边分别手动启动时

将上面命令的 `--launcher ssh` 改为 `--launcher manual`。第一台机器的终端会打印两条 `HYDRA_LAUNCH:` 命令，并等待代理连接。

保持该终端运行。在两台机器各打开一个终端，加载 CANN 和 MPI 环境，然后分别执行对应的代理命令。复制时去掉行首的 `HYDRA_LAUNCH:` 标签；`--proxy-id 0` 对应 `HOST0`，`--proxy-id 1` 对应 `HOST1`。代理命令包含本轮实际的 `--control-port HOST0:PORT`，应整行使用本次输出。

```bash
export MPI_HOME=/usr/local/mpich-3.2.1
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$MPI_HOME/lib64:${LD_LIBRARY_PATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 在对应机器执行本轮打印的 hydra_pmi_proxy 完整命令。
```

两条代理命令都连接后，MPI 为它们分配同一个作业中的 rank，随后初始化 SHMEM。手动启动也使用一个 `mpiexec`；分别执行两个 `mpiexec -n 16` 会形成两个独立的 MPI 作业。

`--port-range` 控制 MPICH 可用的端口范围，实际端口由 MPICH 选择。`--job-timeout` 默认 3600 秒，包含等待手动启动的时间。启动前可加 `--dry-run`，检查脚本生成的命令。

## 采样与结果

每种 BS 先运行一次 Dispatch，生成真实的路由表和专家输入。用例把 Dispatch 输出当作恒等专家的输出，使用均匀权重执行 Combine，并检查结果是否回到原输入。正式采样前后各检查一次精度，检查按 BF16 或 FP16 的舍入误差设置容差。

每轮流程如下：

```text
上一轮 aclrtSynchronizeStream 完成
→ MPI_Barrier
→ ACL 起始 Event
→ Combine kernel
→ ACL 结束 Event
→ aclrtSynchronizeStream
```

MPI 同步在计时区间外，确保所有 rank 完成上一轮后再复用双 bank。MPI Barrier 仍可能存在返回时刻的偏差；Event 区间也可能包含 Host 下发造成的空隙。这里报告的是 Event 时延，细分 kernel 的执行与等待可再用 CANN profiling 查看。

`rank-N.csv` 保存每个 rank 的全部采样。rank 0 的 `summary.csv` 按同轮最慢 rank 汇总，再计算均值、最小值、最大值、p50 和 p99，单位为 μs。汇总也记录实际核数和三个实验开关。双机输出分布在各自主机，完整采样需要保留两边目录。

全部采样、精度检查和 SHMEM finalize 成功后，各 rank 写出 `rank-N.PASS`。单 rank 异常会调用 `MPI_Abort`，结束本次 MPI 作业。本用例用于功能冒烟和性能对照，完整的随机权重、路由元数据与独立 CPU golden 检查仍由原 Python 精度用例覆盖。

## 启动参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tokens` | 1,2,4,8,16,32,48,64,72 | 每卡 token 数，可给多个 BS |
| `--hidden` | 3584 | hidden 维度，可设为 7168 |
| `--topk` / `--experts` | 8 / 384 | 每 token 专家数与全局专家数 |
| `--capacity` | 256 | 原算子的每卡最大 token 参数，影响自动去重策略；至少覆盖所有 BS |
| `--dtype` | bf16 | 支持 bf16、fp16 |
| `--route` | random | random 为无重复随机专家，hotspot 集中命中前 TopK 个专家 |
| `--seed` | 0 | 路由随机种子 |
| `--warmup` / `--iters` | 10 / 100 | 预热轮数与采样轮数 |
| `--timeout` | 300 | SHMEM 初始化和控制操作超时，单位为秒 |

MPICH 手动启动行为可查阅 [Hydra 3.2.1 实现](https://github.com/pmodels/mpich/blob/v3.2.1/src/pm/hydra/tools/bootstrap/external/external_common_launch.c)；端口范围和作业超时见 [mpiexec 参数说明](https://www.mpich.org/static/docs/v3.2/www1/mpiexec.html)。
