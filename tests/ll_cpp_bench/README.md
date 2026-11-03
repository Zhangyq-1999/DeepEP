# LL Combine C++ 性能用例

## 用途与依赖

本用例直接调用仓库里的 Dispatch 和 Combine kernel，支持单机 EP2、EP4、EP8、EP16，以及双机 EP8、EP32。MPI 负责 SHMEM 初始化、Host 同步和结果汇总，SHMEM 负责设备通信。编译和运行都使用 C++、CMake 与 Shell，无需安装 Python 包或复制 Conda 环境。

两台机器需要相同的源码、编译选项和 SDK：

- Ascend950、CANN 9.2.0 或 9.3.0，以及 C++17 编译器和 CMake 3.20 以上版本。
- `/usr/local/mpich-3.2.1`。用例与 SHMEM 的 MPI 插件需使用这套 MPI。
- 当前分支配套的 HNS_1825 SHMEM SDK。`SHMEM_ROOT` 指向包含 `include`、`lib` 和 `deepep-sdk.json` 的安装目录，`lib` 下需有 `aclshmem_bootstrap_mpi.so`。

可以复用此前编译 LL 算子时生成的 SHMEM SDK。这个 SDK 独立于 Conda。没有现成 SDK 时，下面的 Shell/CMake 脚本会编译仓库锁定的 SHMEM 子模块，并登记 SDK 信息。

一次运行中的所有版本共用一个 MPI 作业和一套 ACL、SHMEM 资源。程序初始化一次，按顺序完成各版本测试，最后统一释放资源。切换版本前，所有 rank 等待上一版结束，再清零测试工作区。清理、输入准备和预热均在 Event 计时之外。

各版本使用独立编译的 kernel，实验开关只进入实验版。base 保留冻结的上游 Combine 源码；每机只选 4 张卡时，仅将其公共头文件中的每机参与 rank 数从 16 改为 4，使基线与实验版使用相同拓扑。每版都单独检查精度、采样并输出结果。

## 一次编译并运行当前实验

已有配套 SHMEM SDK 时，两台机器各执行以下命令。脚本默认使用 `/usr/local/Ascend/cann-9.2.T101`、`/usr/local/mpich-3.2.1` 和当前仓库的 `third-party/shmem/install/shmem`。已设置的 `ASCEND_HOME_PATH`、`MPI_HOME`、`SHMEM_ROOT` 优先于这些路径。脚本自动加载所选 CANN 的 `set_env.sh`，并配置 MPI、SHMEM 的库路径。

```bash
git fetch origin
git switch perf/ll-combine-small-batch
git pull --ff-only origin perf/ll-combine-small-batch
bash tests/ll_cpp_bench/build_all.sh
```

`build_all.sh` 默认更新以下 8 个构建目录，每个构建使用 4 个编译任务。Dispatch 用于准备 Combine 的输入和路由，同一配置只编译一次，各版本复用它的目标文件。每个版本再单独编译 Combine，并链接成独立的 kernel 库。首次完整编译的设备编译单元从 144 个减少到 25 个，实际耗时仍取决于机器和编译器。

已有目录继续用于增量编译。只修改 Combine 时，Dispatch 无需重编；修改 Dispatch 或它依赖的头文件时，构建系统自动重编受影响的部分。共享目标文件保存在 `build/ll-cpp-dispatch`，不同编译器、SDK 和每机 rank 数使用独立目录。更新本次构建和超时功能后，两台机器都执行一次 `build_all.sh`，更新所有选中的版本。

| 构建名 | 启用的实验 | 打点 |
| --- | --- | --- |
| `base` | 冻结的上游原版 | 关闭 |
| `issue` | 在准备下一包前提交上一包 RDMA 回传 | 关闭 |
| `probe` | 记录已确认的回传分量，只检查尚未确认的有效分量 | 关闭 |
| `issue-probe` | 同时启用以上两项实验 | 关闭 |
| `ref-profile` | 实验开关全部关闭，用作打点对照 | 开启 |
| `issue-profile` | 提前提交回传 | 开启 |
| `probe-profile` | 记录就绪检查进度 | 开启 |
| `issue-probe-profile` | 同时启用以上两项实验 | 开启 |

此前的 `remote`、`pipeline`、`return` 及对应打点版，以及任务缓存 `opt-profile`、分核 `r36-profile`、提前归约 `early-profile` 均改为按需编译、运行。已有构建和结果文件继续保留。需要复测时，用 `--variants` 指定对应名称。

默认使用当前两台机器：节点 0 为 `90.90.93.84`，节点 1 为 `90.90.93.184`。**每台使用后 4 张卡，即设备 12、13、14、15，共 EP8。** `build_all.sh` 和 `run_all.sh` 默认均为 `--ranks-per-host 4`。这一参数控制参与通信的 rank 分组，设备编号由 `--devices` 独立指定。脚本根据本机 IP 识别节点。两边编译完成后，第一台执行：

```bash
bash tests/ll_cpp_bench/run_all.sh
```

第一台会打印新的 `RUN_ID` 和第二台的完整命令。第二台复制该命令即可：

```bash
bash tests/ll_cpp_bench/run_all.sh 第一台打印的RUN_ID
```

默认先跑 4 个干净版本，再跑 4 个打点版本。每版测试 **BS32、48、64、72、256**，形状为 EP8、H3584、TopK 8、384 个专家、BF16、随机路由、seed 0、capacity 256。每个 BS 预热 10 轮；干净版采样 100 轮，打点版采样 10 轮。`--iters` 和 `--profile-iters` 分别控制两类采样次数。

干净版在编译时关闭打点，使用独立 kernel。打点版在预热前启用所选打点级别，让预热和采样执行相同路径。一次运行可同时包含两类版本，`--profile 2` 只作用于支持打点的构建。

每次启动都会生成新的结果目录，保留此前数据。两台机器分别显示本机进度，包含 SHMEM 初始化、输入准备、Dispatch、Combine 精度检查、预热、采样和资源释放。预热和采样每 10 轮更新一次完成数。状态没有变化时，脚本每 15 秒打印当前阶段和等待时间；这表示尚未收到新进度，不能单凭它判断卡死。

进度文件保存在 `node-N/progress.txt`，由本机第一个 rank 更新。更新发生在 Event 计时之外、下一次 MPI Barrier 之前，不向设备 kernel 增加打点。第一台在测试结束后按 BS 打印对比表和诊断数据；分析时通常提供第一台的 `summary.txt` 即可。

每次设备 stream 等待默认最多 10 秒，可用 `run_all.sh --kernel-timeout 20` 改为 20 秒。超时或设备错误会打印 rank、设备、BS、轮次和阶段，然后调用 `MPI_Abort` 终止本次作业，不继续采样或写入 PASS。该限制使用 [CANN 的带超时 stream 同步接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0077.html)，不修改原版 kernel。SHMEM 初始化仍使用独立的 `--timeout` 参数，MPI 作业总时限由 `--job-timeout` 控制。

`issue` 在读取下一条路由后、准备下一包数据前调用上一包的 Put，避免把下一包的向量计算等待排到上一包 WQE 搬运之前。它沿用原暂存区、输出槽和 CQ 回收规则。

`probe` 为每个 token 保存检查进度，跳过无效 slot 和已确认的分量。每个分量仍检查全部块尾标记，全部有效分量到齐后才按原顺序归约。它减少重复标记搬运，但可能增加单个 token 的检查次数，因此单独保留一个实验版本。BS256 同样测试这两项改动。

这些改动尚未证明能稳定降低设备时延。先看干净版 Event，再用打点版解释差异。入口仍只执行 Host MPI Barrier，它能减小提交偏斜，但不保证各卡 kernel 同时开始。

参数仍可覆盖，例如：

```bash
# 只重编本轮关心的两个版本，也可覆盖 SDK 路径和并行度。
bash tests/ll_cpp_bench/build_all.sh \
    --variants ref-profile,issue-profile --jobs 8 --shmem-root "$SHMEM_ROOT"

# 单机 EP16：先把全部选中版本重编为每机 16 个 rank。
bash tests/ll_cpp_bench/build_all.sh --ranks-per-host 16
bash tests/ll_cpp_bench/run_all.sh --ep 16 --ranks-per-host 16
# 双机 EP32 使用同一套 16-rank 构建，第二台复制第一台打印的命令。
bash tests/ll_cpp_bench/run_all.sh --ep 32 --ranks-per-host 16

# 上游 base 没有 EP2 特化，EP2 使用实验 kernel。
bash tests/ll_cpp_bench/build_all.sh --variants issue,issue-profile
bash tests/ll_cpp_bench/run_all.sh --ep 2 --variants issue,issue-profile

# 单机 EP4 指定设备，覆盖 BS、预热和采样次数。
bash tests/ll_cpp_bench/run_all.sh --ep 4 --devices 6,7,14,15 \
    --variants ref-profile,issue-profile --tokens 32,72,256 --warmup 10 --profile-iters 20

# 更换双机地址、端口或版本顺序时，第二台复制此命令打印的加入命令。
bash tests/ll_cpp_bench/run_all.sh --hosts HOST0,HOST1 --node-rank 0 \
    --master-port 30071 --variants issue-profile,ref-profile
```

编译与运行的 `--ranks-per-host` 必须一致。切回默认双机 EP8 时，两边重新执行不带参数的 `build_all.sh`。`launch.sh` 会在初始化前拒绝拓扑不匹配的旧构建。

用 `--dry-run` 可以查看编译或启动命令。识别不到本机 IP 时，使用 `--node-rank 0` 或 `--node-rank 1` 指定节点。首次准备 SDK、定制编译选项和底层启动方式见下文。

## 拉代码和编译

### ops-transformer 的 SHMEM 适配基线

`ops-shmem-base` 使用 [ubx_reserve 的 Combine 实现](https://gitcode.com/zzg_code/ops-transformer/blob/e6c7d2e57dd222e5a774ba3c89392a50a6a26a78/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h)，固定到提交 `e6c7d2e5`。它保留原实现的分核、机内汇集、部分归约、跨机回传和最终归约流程。通信上下文由 SHMEM 内存和 PE 映射组成，测试侧继续使用 MPI 和 SHMEM 初始化，直接启动 kernel。

跨机适配使用 `aclshmemx_roce_qp_put_nbi`，在原有批次边界调用 `aclshmemx_roce_qp_quiet`。这与原仓 Hcomm 的批量提交、CQE 策略有差异，结果用于比较 SHMEM 后端上的算法方案。单机也运行这套分层算法，只执行机内阶段。原来的 `base` 继续使用冻结的 DeepEP kernel。

两边分别按需编译，默认 `build_all.sh` 构建前文的 8 个版本：

```bash
bash tests/ll_cpp_bench/build_all.sh --ranks-per-host 16 --variants base,ops-shmem-base
```

首次构建会下载固定版本的 kernel 及依赖，保留上游许可证。已有该版本的干净 checkout 时，可用 `OPS_SOURCE_DIR=/path/to/ops-transformer` 指定源码目录。新基线只编译自身 kernel，不编译 DeepEP Dispatch 的多种特化。

第一台执行以下命令，第二台复制第一台打印的加入命令：

```bash
bash tests/ll_cpp_bench/run_all.sh --ep 32 --ranks-per-host 16 --variants base,ops-shmem-base --profile 0 --iters 100
# 单机比较：把上面的 --ep 32 改为 --ep 16。
```

这份适配基线目前按每机 16 个 rank 构建，尚未接入两机各 4 卡的拓扑。初版支持 H3584/H7168、TopK 8、384 个专家、BF16/FP16，BS 和设备选择沿用现有参数。测试在 CPU 上根据相同路由生成 expert 输入、行索引、计数和位图，准备工作在计时外完成。每个 rank 检查输出，汇总仍使用逐轮 rank-MAX Event。构建目录的 `ops-baseline.txt` 和结果 CSV 记录源码版本及通信适配方式。

这份基线关闭算子内打点，使用 `--profile 0`。Host 检查覆盖路由与汇集位图的一致性、构建选择及源码适配范围；CANN 编译和 NPU 精度、性能仍需在目标机器验证。

### DeepEP 基线与实验版

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

`build.sh` 编译统一入口 `ll_combine_bench`、版本模块 `libll_bench_variant.so` 和 LL kernel 库。程序按模块路径加载各版本。每个构建目录生成不同名称的 kernel 库，避免同名函数串用。

CANN 的 Device 编译器与 Host C++ 编译器使用各自的构建目录。构建结果保存在 `build/ll-cpp-base`，其中 `bench-build.txt` 记录源码摘要和实验开关。

如果已有其他目录下的配套 SDK，可以跳过 `prepare_shmem.sh`，直接设置实际的 `SHMEM_ROOT`。脚本会保留已有构建；如果子模块中存在尚未登记的 `build/install`，它会停止并提示使用现成 SDK 或新 checkout。

base 和 opt 使用各自的 Combine kernel 源码。base 从上游提交 `5b38bf2ea1f67f38c0c2d3662618ae1a0457bfc6` 提取原始 kernel 及其依赖头文件，保存在构建目录中。Combine 算法文件保持不变。每机 4 个 rank 的构建只替换 `ll_group_policy.h` 中的每机 rank 数常量；16-rank 构建的全部原始文件保持不变。实验开关和打点只进入 opt。`bench-build.txt` 记录实际源码路径、原版提交和编译选项。

`--variant base` 明确选择原版；`--variant opt` 选择实验版。省略参数时，脚本根据实验开关选择：全部关闭时使用 base，启用任意实验或打点时使用 opt。各优化开关可以单独编译：

| 编译参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--ranks-per-host 4` | 16 | 编译每机参与 rank 数；便捷脚本 `build_all.sh` 默认传 4 |
| `--early-issue ON` | OFF | 在下一包准备前提交上一包，缩短回传推迟时间 |
| `--progress-probe ON` | OFF | 记录已到齐的贡献，后续只检查尚未确认的有效 slot |
| `--small-batch ON` | OFF | 缓存 FIRST_HOP 任务列表，减少重复扫表和空闲缓冲槽处理 |
| `--recv-cores 8` | 0 | 指定接收、汇集和最终归约核数；0 沿用核数均分 |
| `--early-reduce ON` | OFF | BS ≤ 72 时，按原 slot 顺序归约已到齐的贡献，同时推进至多两个 token |
| `--remote-first ON` | OFF | 跨机结果优先回传，本机结果随后处理 |
| `--defer-return ON` | OFF | 多输出槽时，用下一包准备掩盖上一包 GM 暂存；组内末包立即提交 |

回传顺序与暂存流水的分析和新增打点见 [回传优化对照](RETURN_PIPELINE_AB.md)。这两个开关默认关闭，先独立对比，再测试组合。

显式接收核数支持 8～36，并须小于实际 AIV 总核数。其余核负责 DIRECT 回传，TERMINAL 暂存仍由全部核参与。各 rank 使用相同的核数配置；最后 8 核负责组内就绪通知。

分核调整参考了上游 [ll-configurable-groups 的实现](https://github.com/TangYunfei147/DeepEP-LL/blob/705decf9bd24a0ead24368a2a19307c36dd09ea2/tests/ll_stage_timer/CORE_SPLIT.md)。接收核可以超过总核数的一半，例如 56 核时配置 20 个发送核和 36 个接收归约核。当前分支继续使用已有控制区容量、Get 汇集和跨机 RDMA 回传，base 使用原冻结源码。

例如，构建一个组合实验版：

```bash
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --build-dir "$PWD/build/ll-cpp-opt" \
    --small-batch ON --recv-cores 8 --early-reduce ON
```

分别构建原版、单项优化版和组合版，再用相同参数对比。正式性能构建使用 `--profile OFF`，编译器会移除诊断代码。打点用于定位问题，启用打点后的 Event 耗时单独统计。

只对比分核时，可以构建 `r36-profile`。它关闭其他优化，只把接收核数改为 36：

```bash
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --mpi-home "$MPI_HOME" \
    --build-dir "$PWD/build/ll-cpp-r36-profile" --variant opt \
    --small-batch OFF --recv-cores 36 --early-reduce OFF --profile ON
```

将它与下文的 `ref-profile` 用 `--variants ref-profile,r36-profile` 对比，运行时保持 `--profile 2`。双机测试需要两边都编译。56 核时，原流程使用 28 个接收核，实验版使用 36 个；重点比较 DIRECT 回传和最终归约的收尾。分核调整已通过 Host 检查，设备精度和时延仍需实测。

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

脚本用第一个版本目录中的 `ll_combine_bench` 启动一次 MPI 作业。入口加载各目录的 `libll_bench_variant.so`，随后按指定顺序测试。`base,opt` 会先测 base，再测 opt，两版共用同一次 SHMEM 初始化。也可以用 `--variants base,opt,base` 做 A/B/A 对照。每次切换都会清理共享的数据和标记，保留 SHMEM 已建立的通信资源。

单机测试每次都会在 `results/ll-cpp/<run-id>/` 下创建带时间戳和随机后缀的子目录。同一条命令可以重复执行，之前的 CSV、日志和汇总都会保留。脚本开头的 `Results:` 显示本次目录。`--variants` 也支持自定义构建名：例如 `ref-profile` 对应 `build/ll-cpp-ref-profile/libll_bench_variant.so`。

EP2 使用新增的 G2 Plain16 特化，支持 BF16 和 FP16；EP4、EP8、EP16 使用 G4。单机数据走 MTE。双机 EP8 用 G4，双机 EP32 用 G16，跨机回传都走 RDMA；FIRST_HOP 汇集保留机内 Get 方案。以下 `build.sh`、`launch.sh` 命令默认每机 16 个 rank；双机 EP8 则给两者都传 `--ranks-per-host 4`。

小 EP 验证覆盖本机通信和归约；跨机 RDMA 与对应中转需用双机 EP8 或 EP32 验证。本次小 EP 入口在 C++ 用例中提供，原 Python 用例的 EP 范围保持不变。

## 双机 32 卡

**两台机器各执行一次 `launch.sh`，共同组成一个 32-rank MPI 作业。** 脚本封装 MPICH 3.2.1 的 Hydra 启动流程，自动启动本机代理和 16 个 rank。运行时无需 SSH，也无需复制代理命令。

两台机器先加载前文的 CANN、MPI 和 SHMEM 环境。两边使用相同的仓库绝对路径、编译结果、`--hosts`、`--master-port`、`--variants` 和 `--run-id`。双机继续使用共同指定的结果目录，每轮需在两边填写相同的新 `--run-id`。下面的 `HOST0`、`HOST1` 替换成实际 IP。

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

`--master-port` 指定本次 MPI 作业的控制端口，所有版本共用这个端口。MPI 数据连接使用其正常端口分配；两台机器需要允许 MPI 通信。SHMEM 通过 MPI bootstrap 初始化一次。

`--join-timeout` 默认 300 秒，`--job-timeout` 默认 3600 秒。脚本在超时或失败时退出，并清理本次启动的进程。第一台的退出码反映 MPI 作业的整体状态，各节点同时检查本机 rank 的 PASS 文件。启动前可加 `--dry-run` 查看命令。

`run.sh` 保留为单个 MPI 作业的底层启动入口。常规单机与双机测试使用上面的 `launch.sh` 即可。

MPICH 3.2.1 的内置 `manual` 模式在关闭 autofork 后，可能在打印 `HYDRA_LAUNCH_END` 后报 `free(): invalid pointer`。当前脚本通过 `-launcher-exec` 调用仓库中的 `hydra_proxy_command.sh`，由它打印代理命令，再由每台机器的启动脚本运行本机代理。这样可以保留手动会合方式，系统 MPI 和 kernel 构建继续复用。该辅助脚本只打印命令，不建立 SSH 连接。

所有版本的运行日志保存在 `node-N/suite.log`，第一台另有 `node-0/suite.log.controller`。如果 controller 异常退出，脚本会同时打印这两份日志。优先查看 controller 的首个错误；随后出现的 `Connection refused` 可能只是 controller 退出后的结果。

Linux 上，`launch.sh` 会将当前进程的文件描述符软上限提高到至少 65,536，并打印软、硬上限。每台机器的 MPI 代理和测试进程会继承这个额度。脚本保留原硬上限；硬上限不足时，先退出并提示调整启动环境。此设置只影响本次脚本及其子进程。

EP32 初始化时若出现 `Too many open files`，随后报 `HrtRaBlockGetSockets` 超时，应先检查这个额度。已有的用例可在两边终端执行 `ulimit -Sn 65536` 后重试，无需重新编译。提高额度后仍有超时时，再检查设备网络和 SHMEM 建链日志。

## 采样与结果

每种 BS 先运行一次 Dispatch，生成真实的路由表和专家输入。用例把 Dispatch 输出当作恒等专家的输出，使用均匀权重执行 Combine，并检查结果是否回到原输入。正式采样前后各检查一次精度，检查按 BF16 或 FP16 的舍入误差设置容差。

所有 EP 默认使用 Host MPI 同步。每轮先确认上一轮的设备任务完成，再通过 `MPI_Barrier` 等待所有 rank。Barrier 返回后，各 rank 提交以下任务：

```text
起始 Event → Combine → 结束 Event
```

预热也在每轮提交前执行 MPI Barrier。用例不再插入设备门控或 SHMEM 入口同步 kernel，原版 Combine 内部保持不变。MPI 同步位于 Event 计时之外，只协调 Host 提交；各卡仍可能存在设备启动偏差，分析时需要结合 CANN profiling 确认。

`--entry-sync host` 是默认配置，`auto` 同样选择 Host MPI 同步。旧的 `device` 选项已移除。

`rank-N.csv` 保存每个 rank 的全部采样。rank 0 的 `rank-max.csv` 保留每轮最慢 rank 的编号和 Event 耗时；`summary.csv` 按同轮最慢 rank 汇总，再计算均值、最小值、最大值、p50 和 p99，单位为 μs。汇总也记录实际核数、原版标识、启动同步方式、实验开关和打点状态。双机输出分布在各自主机，完整采样需要保留两边目录。

单机 `--run-id ep4-ab --variants base,opt` 的结果目录形如 `results/ll-cpp/ep4-ab/20261102-153000-Ab12Cd/`，其中 `0-base`、`1-opt` 保存两版结果，`node-0` 保存启动日志。双机仍使用 `results/ll-cpp/<run-id>/`，其中 `node-0`、`node-1` 保存各节点的日志。

每版通过精度检查和采样后，各 rank 在该版目录写出 `rank-N.SAMPLED`。全部版本完成且共享 SHMEM finalize 成功后，程序才为各版写出 `rank-N.PASS`。后续版本或释放过程失败时，已完成的采样文件继续保留，整次运行记为失败。

单 rank 异常会调用 `MPI_Abort`，结束本次 MPI 作业。本用例用于功能冒烟和性能对照，完整的随机权重、路由元数据与独立 CPU golden 检查仍由原 Python 精度用例覆盖。

## 分阶段打点

打点仅在 opt 构建中提供。编译时加 `--profile ON`，运行时用 `--profile 1` 记录阶段边界，或用 `--profile 2` 进一步累计热循环耗时和轮询次数。

### 原流程与优化流程对照

原版 base 保持独立、无打点。细粒度对照使用两份实验源码构建：`ref-profile` 关闭全部优化，记录原流程；`opt-profile` 开启 `small_batch`，记录该优化的影响。两份都保持默认分核、关闭 `early_reduce`，并启用相同的二级打点。构建目录独立，原有 base 和 opt 构建继续保留。

```bash
bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --mpi-home "$MPI_HOME" \
    --build-dir "$PWD/build/ll-cpp-ref-profile" --variant opt \
    --small-batch OFF --recv-cores 0 --early-reduce OFF --profile ON

bash tests/ll_cpp_bench/build.sh \
    --shmem-root "$SHMEM_ROOT" --mpi-home "$MPI_HOME" \
    --build-dir "$PWD/build/ll-cpp-opt-profile" --variant opt \
    --small-batch ON --recv-cores 0 --early-reduce OFF --profile ON

bash tests/ll_cpp_bench/launch.sh \
    --ep 4 --variants ref-profile,opt-profile --run-id ep4-profile-ab -- \
    --devices 6,7,14,15 --tokens 32,48,64,72 \
    --hidden 3584 --topk 8 --experts 384 --capacity 256 --dtype bf16 \
    --route random --seed 0 --entry-sync host \
    --warmup 10 --profile-iters 10 --profile 2
```

以上启动命令可以直接重复执行。每次的新目录内，`0-ref-profile` 和 `1-opt-profile` 分别保存原流程与优化流程的诊断数据。需要 A/B/A 对照时，使用 `--variants ref-profile,opt-profile,ref-profile`。只看阶段边界时，把运行参数改为 `--profile 1`。

`ref-profile` 用于阶段分析，最终原版性能仍以冻结源码的 base 构建为准。分析其他优化时，每次只调整一个编译开关。确认效果后，以 `--profile OFF` 编译，再比较两版的完整 Event 耗时。

### 在终端查看汇总

测试结束后，第一台打印每个 BS 的对比表，包含均值、P50、最小值、P95、最大值和变异系数（CV）。`clean` 行用于性能比较，`profile` 行用于定位耗时。当前报告用线性插值计算分位数；选取诊断轮次时仍选择排序后中间偏前的真实样本。

报告保留全部样本。`WIDE` 表示 `(最大值 − 最小值) / P50 > 20%`，提醒检查波动；`IQR` 列出超过 `Q3 + 1.5 × (Q3 − Q1)` 的高值候选。这些标记用于排查，程序继续用全部样本计算统计值。

打点回显中的 `E` 行给出逐轮最慢 rank 与耗时。`D` 行标识全局中位轮和最慢轮，`S`、`R` 分别展示该轮最慢 rank 的发送核与接收核。`P` 行列出 Put 返回相对本核入口较晚的候选核。各核时间保留 tick；分析跨核、跨 rank 的绝对先后需要时钟校准。

rank 0 在采样后汇总选中轮次的全局代表记录，保存在各版本目录的 `diagnostic.csv`。全量逐核记录仍保存在各卡所属机器的 `rank-N-profile.csv`。报告保存为运行目录下的 `summary.txt`，可以重新生成：

```bash
bash tests/ll_cpp_bench/summarize.sh --suite results/ll-cpp/RUN_ID
```

需要完整的本地阶段汇总时，使用原有入口：

```bash
bash tests/ll_cpp_bench/summarize.sh results/ll-cpp/RUN_ID/4-ref-profile
```

这两个命令只运行 Host C++ 汇总程序，无需 CANN、MPI 或 Python。逐阶段最大值可能来自不同核，需分别分析。所有核内偏移以本核入口为零点。
### 字段与分析方法

每个 rank 的 `rank-N-profile.csv` 保存每轮、每个 AIV 核的记录。一级打点记录阶段边界和任务量。二级打点进一步区分以下操作：

| 阶段 | 可以定位的步骤 | 主要字段前缀 |
| --- | --- | --- |
| 初始化 | 读取有效行数、计算工作区布局、分配核内缓冲 | `init_prefix`、`init_count_end`、`init_layout_end` |
| TERMINAL 暂存 | 读取类型表、筛选行、读取路由、等待缓冲复用、搬入数据、写入共享区和等待写完 | `stage_select_*`、`stage_metadata`、`stage_reuse_wait`、`stage_input`、`stage_store_issue`、`stage_drain` |
| 就绪通知 | 等待本卡所有核准备完成、清理标记、发布组内通知 | `notify_poll_*`、`notify_reset`、`notify_publish` |
| 组内等待 | 读取各卡 ready、检查标记、清理 ready | `ready_poll_*`、`ready_reset` |
| DIRECT 回传 | 读取路由、筛选目标、读取输入、等待输出槽、打包、提交回传和收尾 | `direct_route_load`、`direct_scan`、`direct_input_*`、`direct_output_wait`、`direct_pack`、`direct_submit`、`direct_drain` |
| FIRST_HOP 汇集 | 建立任务表、扫表、读取任务元信息、本地读取、Get 其他卡的数据、部分归约、打包和回传 | `plan_*`、`partial_scan_*`、`partial_metadata`、`partial_local_*`、`partial_get_*`、`partial_compute_issue`、`partial_pack`、`partial_submit` |
| 最终归约 | 读取 mask 和 weight、建立待处理列表、检查数据到齐、搬入贡献、归约、转换输出格式和写回 | `final_meta_load`、`final_pending_build`、`probe_*`、`final_input_*`、`final_compute_issue`、`final_output_pack`、`final_store_issue` |
| 跨机 RDMA 回传 | 将 UB 数据暂存到 GM、等待暂存完成、调用 Put、quiet 回收 | `rdma_stage_issue`、`rdma_stage_wait`、`rdma_put`、`rdma_quiet` |

表中的累计耗时字段以 `_ticks` 结尾。`*_calls` 记录调用次数，`*_misses` 记录未就绪次数。`stage_rows`、`direct_rows`、`partial_rows`、`final_tokens` 和 `contributions` 记录本核工作量。`partial_local_contrib` 记录 FIRST_HOP 自身专家行，`partial_get_contrib` 记录按路由从共享区拉取的贡献。EP4 没有跨组 FIRST_HOP 任务，汇集和 RDMA 数据量可能为零。

`notify_local_ready` 分开本卡准备等待与通知发布。`direct_first_submit`、`direct_last_submit` 和 `direct_pending_end` 标出首个提交调用、末个调用返回以及待提交任务处理完的位置。`final_first_ready`、`final_last_ready` 标出成功检查数据就绪的位置；`final_last_store` 标出末个输出写回的下发位置。最终输出完成仍以原有 drain 和 kernel 收尾为准。

时间字段使用设备原始 tick。`entry_tick` 保留原始值，其余阶段时间戳均减去本核入口时间；缺失阶段留空。`profile_init_end` 标出诊断记录初始化完成的位置，可据此分开打点初始化与业务初始化。一级打点中的二级字段留空。schema 4 每核使用 1536 B 记录，在原有收尾完成后集中写出。

分析时先选择同一配置、同一轮的记录，比较发送核与接收核的任务量和阶段耗时。常见判断如下：

- `notify_begin` 到 `notify_local_ready` 较长：检查各核的暂存结束时间。`notify_publish_ticks` 较长则关注通知发布本身。
- `probe_misses` 和 `probe_miss_ticks` 较大：接收核多次检查时数据尚未齐备，需要结合发送侧提交和收尾分析。`probe_load_ticks` 包括标记加载及其等待，`probe_check_ticks` 包括 `ReduceSum` 及原有 `V_S` 等待。
- `direct_submit_ticks` 较大：先看是否跨机。跨机时继续区分 GM 暂存、Put 调用和 quiet，查看 `rdma_*` 字段；机内则观察提交、输出槽等待与最终 drain。
- 就绪检查开销较小、归约仍较慢：检查 `final_input_wait_ticks`、`final_compute_issue_ticks` 和 `final_output_pack_ticks`。它们分别反映已有输入等待、计算指令下发和输出转换到可交给 MTE3 的过程。

`early_issue_ticks` 记录提前提交上一包所花的时间，其中可能包含 `rdma_stage_wait_ticks` 和 `rdma_put_ticks`，这些字段不能相加。`probe_flag_blocks` 记录实际加载的标记块数，`probe_confirmed_slots` 记录已确认的分量数；`probe_blocked_token/slot` 记录最后一次未就绪的位置。结合收尾时间判断是否减少了重复检查，不能仅凭计数下降认定算子更快。

单次回传调用的最大耗时保存在 `direct_submit_max_ticks` 或 `partial_submit_max_ticks`，并记录对应 peer 和 token。`probe_max_ticks` 记录最慢的一次标记检查及其 token、slot；整 token 检查时 slot 为 `4294967295`。该值反映一次检查的耗时，持续未就绪的情况结合累计次数和耗时判断。`rdma_quiet_max_peer` 标出单次 quiet 等待最久的目标。

计时沿用原有等待指令，记录标量流水执行到各打点位置的时间。`*_issue_ticks` 表示下发指令所花的时间。`MTE2_V`、`MTE3_V` 等事件约束向量流水，标量打点可能继续向前执行，因此这些 `*_wait_ticks` 只能反映对应调用区间。后续 `V_S` 或 `PIPE_ALL` 可能承担前序操作尚未完成的等待。例如，`probe_check_ticks` 包含 `ReduceSum` 和 `V_S`，也可能包含前面标记搬入的剩余等待。

外层阶段耗时包含内层细分项，例如 `direct_submit_ticks` 可能包含 `rdma_*` 耗时，应分别阅读。不同核的并行耗时也分别分析。跨设备时间戳需要先确认时钟对齐，tick 换算为时间需要确认计数器频率。

诊断记录和导出会影响时延。性能对比使用同轮测试中对应的干净版本；自行构建时，用 `build.sh --profile OFF` 生成无打点 kernel。

## 启动参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tokens` | 32,48,64,72,256 | 每卡 token 数，可给多个 BS |
| `--hidden` | 3584 | hidden 维度，可设为 7168 |
| `--topk` / `--experts` | 8 / 384 | 每 token 专家数与全局专家数 |
| `--capacity` | 256 | 原算子的每卡最大 token 参数，影响自动去重策略；至少覆盖所有 BS |
| `--dtype` | bf16 | 支持 bf16、fp16 |
| `--route` | random | random 为无重复随机专家，hotspot 集中命中前 TopK 个专家 |
| `--seed` | 0 | 路由随机种子 |
| `--warmup` / `--iters` | 10 / 100 | 预热轮数与无打点运行的采样轮数 |
| `--profile-iters` | 10 | 启用打点时的采样轮数 |
| `--entry-sync` | host | 每轮提交前执行 Host MPI Barrier；auto 同样选择 host |
| `--profile` | 0 | 0 关闭，1 记录阶段边界，2 增加热循环打点；干净构建自动保持关闭 |
| `--timeout` | 300 | SHMEM 初始化和控制操作超时，单位为秒 |
| `--kernel-timeout` | 10 | 单次设备 stream 等待上限，单位为秒；包含准备 Dispatch 和采样 Combine |

上表是可执行文件和 `launch.sh` 的默认值。便捷入口 `run_all.sh` 默认使用 `--profile 2`，干净版采样 100 轮，打点版采样 10 轮；其余形状参数相同。

MPICH 手动启动行为可查阅 [Hydra 3.2.1 实现](https://github.com/pmodels/mpich/blob/v3.2.1/src/pm/hydra/tools/bootstrap/external/external_common_launch.c)；端口范围和作业超时见 [mpiexec 参数说明](https://www.mpich.org/static/docs/v3.2/www1/mpiexec.html)。
