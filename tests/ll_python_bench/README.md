# Python Combine 性能测试

## 测试内容

这套脚本用 PyTorch 启动多机测试，比较算子前同步对 Combine 时延的影响。默认使用 `py311_zyq` 环境，两台机器各用设备 `12,13,14,15`，总共 EP8。

| 参数 | 默认值 |
| --- | --- |
| 主机 | `90.90.93.84,90.90.93.184` |
| 算子 | `base,input6,input6-l2,input6-remote,input6-l2-remote`，各有独立打点版 |
| BS | `32,48,64,72,256` |
| Hidden / TopK / Experts | `3584 / 8 / 384` |
| 数据和路由 | BF16、random、seed 0 |
| Capacity | 256 |
| 预热 / 干净版采样 / 打点版采样 | 10 / 100 / 10 次 |
| 算子前同步 | SHMEM 混合 MTE/RoCE 设备同步，计时不包含同步 |

Python 流程复用 C++ 用例的输入、路由、精度检查和原始 Combine kernel。Dispatch 只负责准备 Combine 的输入，放在计时外。每个 worker 只初始化一次 NPU、SHMEM；使用 HCCL 时也只创建一次 HCCL 通信组，依次完成所有版本的全部 BS。

## 两台机器分别编译安装

在现有仓库目录执行：

```bash
git pull --ff-only https://github.com/Zhangyq-1999/DeepEP.git perf/ll-combine-small-batch

bash tests/ll_python_bench/build_all.sh
```

脚本自动激活 `py311_zyq`，用该环境的 Python 编译并安装测试扩展。扩展、对应 kernel 库和构建记录安装在 `py311_zyq/lib/python3.11/site-packages/deepep_ll_bench/base/`。运行脚本自动切换到同一环境，并加载这里的安装产物。

默认编译五个干净版本及五个独立打点版本，复用已有 Dispatch 编译产物。`--variants base` 缩减为 base/ref-profile；编译和运行同时传 `--profile 0` 则只测干净版。脚本保留环境中的 Torch 和 torch-npu 版本，只在缺少构建工具时安装 pybind11 或 CMake。测试扩展独立安装，环境中原有的 `deep_ep` 包保持不变。

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

脚本按 `--devices` 显式绑定各进程，并向 native 传入相同的设备列表。启动时会清除 `ASCEND_RT_VISIBLE_DEVICES` 等可见设备变量，避免设备重编号干扰 HCCL 的拓扑查询。默认仍只在每台的 12～15 卡上执行任务。

建议先对同一份 kernel 做同步方式对照。第一台执行以下命令，再在第二台粘贴它打印的命令：

```bash
bash tests/ll_python_bench/run_all.sh --sync both
```

每个 BS 先测 `host`，再测 `device`，两种模式分别预热。`host` 使用 Gloo 的 Host barrier；`device` 使用与 C++ 用例相同的 SHMEM 设备同步。这个对照保留相同输入和 kernel，但 `host` 模式与 C++ 用例的 MPI 实现并不相同。

## 同步和计时顺序

本地 Notify 控制任务开始执行的时机，提交顺序参考 [Ziha0-Zeng/DeepEP 的提交 b1ed463](https://github.com/Ziha0-Zeng/DeepEP/commit/b1ed463)。跨 rank 同步默认改用 SHMEM 的混合通路方案，参考 [fix_1825_sync 的提交 71a2154](https://gitcode.com/vallenChen/shmem/tree/fix_1825_sync)，实现与 [C++ 用例](../ll_cpp_bench/README.md#采样与结果) 共用。

```text
准备输入、路由和输出（计时外）
    ↓
本地 Notify 暂停目标流
    ↓
提交 SHMEM MTE/RoCE 同步 kernel
    ↓
提交 start Event → Combine kernel → end Event
    ↓
独立控制流释放 Notify
    ↓
SHMEM 同步完成 → start Event → Combine → end Event
```

Notify 保证本机先提交完整的计时任务。SHMEM 同步优先使用目标卡的 MTE 通路，其余目标使用 RoCE，避免向同一 1825 上没有 RDMA 通路的相邻卡发同步消息。正式测试前先单独执行一次同步，回显为 `SHMEM device entry sync preparation`。同步完成时间仍可能存在偏差，需要实测判断改善幅度。

`--sync hccl` 的初始化顺序为：绑定物理设备 → Torch NPU → 创建 HCCL 通信组、首次 all_reduce 并同步完成 → 加载 native 扩展 → 初始化 SHMEM → 再验证一次已存在的 HCCL 通信组。正式采样仍使用 `all_reduce → Work.wait → 排空 Host 提交队列`，同步不计入 Combine Event。

旧流程在 SHMEM 初始化后才创建 HCCL，曾在设备 12–15 的 `libccl_kernel.so / RunAicpuIndOpThreadInit` 报 AICPU 异常。选卡正确，但这份日志不足以证明内部根因。新顺序需要目标机验证，不自动回退到其他同步方式。可独立测试 HCCL，完全不加载 native 或初始化 SHMEM，也不要求先编译扩展：

```bash
bash tests/ll_python_bench/run_all.sh --check-hccl
```

第二台粘贴第一台打印的命令。成功打印 `HCCL_CHECK_PASS`。若这里仍失败，问题发生在 Torch/HCCL 初始化或执行路径，尚未涉及 Combine/SHMEM。启动时 `BIND` 行打印各 rank 的实际设备；`rank-N-runtime.json` 保存选卡、CANN 环境和实际加载的通信库路径，便于继续排查。

启动脚本将 `TORCH_HCCL_BLOCKING_WAIT` 和兼容名称 `HCCL_BLOCKING_WAIT` 设为 `0`。这样 `Work.wait()` 只提交设备流依赖，Host 可以继续提交 Event、kernel 和 Notify 释放操作。[阻塞等待说明](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/collective_communication/TORCH_HCCL_BLOCKING_WAIT.md)解释了两种模式的区别。

可用以下短测检查人为延迟是否进入计时：

```bash
bash tests/ll_python_bench/run_all.sh \
    --tokens 32 --warmup 2 --iters 5 --gate-delay-ms 10
```

该参数只在 rank 0 提交设备同步前增加 10 ms 的 Host 等待。正常情况下，这段等待应留在 Event 计时区间外。此项用于验证计时顺序，不作为正式性能数据。

## 参数和其他实验

默认一次测试五个干净版本和对应的打点版本。只测原版可用 `--variant base`；要测聚合实验，两台机器都编译同一版本，然后在第一台启动，第二台粘贴输出的命令：

```bash
bash tests/ll_python_bench/build_all.sh --variants card-all
bash tests/ll_python_bench/run_all.sh --variant card-all
```

可用 `--variants base,wqe-prefetch` 指定比较列表，或 `--variant base` 只选一个版本。版本各自独立安装、使用独立 kernel 库，原版 kernel 保持不变。版本切换在所有 rank 完成上一阶段后释放 case、清空共享工作区及注册暂存区，再准备下一版输入；SHMEM 通信资源保持到整个测试结束才释放。

当前默认先测五个干净版本，再测相同策略的五个打点版本，只有一次 SHMEM 初始化：

```bash
bash tests/ll_python_bench/build_all.sh
```

默认对照为 `base,input6,input6-l2,input6-remote,input6-l2-remote`。新组合 input6-l2-remote 采用 input6 的 28/28 半区分工，启用 prepare/ready 控制区 L2 和跨机优先，继续立即提交回传。不改变 payload 缓存策略、累加顺序、通知格式、QP 所有权或 CQ 回收。

11 月 6 日 EP32/HCCL 实测 BS32 mean：input6=58.61、input6-l2=56.37、input6-remote=56.85 μs，故新一轮组合这两项。input6-r32、input6-rpipe、input6-rfull 对 BS32 没有稳定优势，移出默认但仍可用 `--variants` 选择。新组合尚待 NPU 实测。

build_all.sh 自动安装到 py311_zyq，两台都需重编；run_all.sh 自动使用同一环境。ID 行保留 layout、control_l2、ready、recv、remote_first。干净版没有 profile 缓冲或核内采样分支；打点版是另一份 Device 库。base 对应 ref-profile，其余对应同名 `-profile`。扩展 suite_api=2，旧扩展会提示重编。

EP32 两台分别编译：

```bash
bash tests/ll_python_bench/build_all.sh --ranks-per-host 16
```

第一台启动所有默认版本，第二台复制它打印的命令。以下显式使用 HCCL all_reduce 作计时前同步，便于核对参考仓；该同步不计入 Combine Event：

```bash
bash tests/ll_python_bench/run_all.sh --ranks-per-host 16 --sync hccl
```

一条命令连续跑全部干净版和打点版，在第二台粘贴打印的命令即可。整个测试共用一个新 RUN_ID。省略 `--ranks-per-host 16` 就是双机各 12–15 卡的 EP8。省略 `--sync hccl` 则统一用默认 SHMEM device 同步，不混用两种同步结果判断算子收益。默认 BS 为 `32,48,64,72,256`，可用 `--tokens` 修改。

IP、端口、设备和 case 参数都可修改：

```bash
bash tests/ll_python_bench/run_all.sh \
    --hosts 90.90.93.84,90.90.93.184 --node-rank 0 \
    --master-port 30011 --devices 12,13,14,15 \
    --tokens 32,48,64,72,256 --warmup 10 --iters 100
```

PyTorch 默认使用端口 30011，SHMEM 使用下一端口 30012。`--shmem-ip-port tcp://IP:PORT` 可单独指定 SHMEM 入口。

跨机启动时，Gloo 默认按 `--hosts` 中**本节点 IP** 查找本机已启用的网卡（使用 `ip -j -4 address show`），在启动 torchrun 前设置 `GLOO_SOCKET_IFNAME`。每台各自选择，不要求四台的网卡名称相同；回显和 `node-N/console.log` 开头记录 `Network: ... GLOO_SOCKET_IFNAME=... source=...`。IP 不属于本机或存在歧义时提前报错，不回退到 loopback。单机保留 PyTorch 默认选择。

优先级为 `--ifname` > 已设置的 `GLOO_SOCKET_IFNAME` > 本节点 IP 自动选择。`--ifname eth0` 同时指定 Gloo/HCCL；仅需覆盖 Gloo 时可在各台分别 `export GLOO_SOCKET_IFNAME=本机网卡`。自动选择只设置 Gloo，HCCL 仍使用自身设置。多接口配置需各进程使用相同接口数量，见 [PyTorch 网络接口说明](https://docs.pytorch.org/docs/2.11/distributed.html#choosing-the-network-interface-to-use)。

`MASTER_ADDR` 只指定会合服务，不能代替 Gloo 全互联的网卡选择。若日志在 `Gloo rendezvous` 报 `connectFullMesh ... remote=[127.0.0.1] ... Connection refused`，错误发生在 NPU/HCCL/SHMEM 初始化前。此前默认依赖主机名解析，换到主机名解析为 loopback 的环境会失败；更新脚本即可，无需重编 kernel。启动修复的本地测试覆盖两机/四机选网与环境传递，真实多机通信仍需目标环境验证。

扩展到每机 16 卡时，两台机器先用 `--ranks-per-host 16` 重编。运行时也传 `--ranks-per-host 16`，默认自动选 0～15 卡；每机 4 rank 时默认选 12～15 卡。`--devices` 仍可覆盖，`--hosts` 支持 1、2 或 4 台机器；跨机参数需保持一致。

## 读取结果

脚本先通过 Gloo 核对各 rank 的测试参数、源码标识和 Python／Torch 版本，通过后才初始化 NPU。Python 只要求主次版本一致，允许 `3.11.17` 和 `3.11.18` 这样的补丁版本差异，并打印提示。脚本仍保存各 rank 的完整版本号。

若其他配置不同，每台机器会打印一段 `LL_PY_MISMATCH_BEGIN` 至 `LL_PY_MISMATCH_END` 的错误信息，列出不同的字段、rank 和实际值。复制这一段即可定位差异。全部 rank 的配置保存在 `rank-0-configuration.json` 等文件中。

每个 BS 完成后立即输出时延，并保存到 `results/ll-python/<RUN_ID>/`。默认带打点时，全部结束后第一台打印 `LL_PY_PROFILE` 后的 `LL_COMBINE` 紧凑报告，再打印配置、版本及 PASS 信息。把这些最终回显发来即可分析全局数据，无需发送 Perfetto 文件。超过容量时自动用 `PASTE_BEGIN/PASTE_END` 分块，每块不超过 62 KiB；每块都要保留。`--profile 0` 时仍打印完整 `LL_PY_REPORT_BEGIN` 至 `LL_PY_REPORT_END`。

汇总先给配置和时延表，再列每个 case 的 `D` 行。表格包含均值、P50、最小值、P99、最大值、末 50 轮 P50、rank0 P50 和最大精度误差。`D` 行补充前 3／10／20 轮均值、各 rank 的 P50 范围，以及最慢 3 轮的轮次、rank 和耗时。轮次从 1 开始；并列最慢时显示编号最小的 rank。`--sync both` 还输出设备同步相对 Host 同步的 P50 降幅。

| 输出 | 内容 |
| --- | --- |
| `report.txt` | 完整时延与分布统计，每完成一个 case 就更新 |
| `profiling-<sync>.txt` | 默认最终打屏的核内紧凑报告，包含 clean/profile 时延对照及全部粘贴分块 |
| `summary.csv` | 每个版本、BS、同步方式的时延统计；`sync_impl` 记录实际同步实现 |
| `<variant>/bs32-device.json` 等 | 配置、所有 rank 的原始样本和汇总 |
| `rank-N.jsonl` | 本 rank 每个 BS 的样本，完成后立即写盘 |
| `node-N/console.log` | 本机回显 |
| `rank-N.PASS` | 精度、采样和 SHMEM 释放均完成后的标记 |
| `<variant>/rank-N.SAMPLED` | 该版本已完成，整个测试尚未必结束 |

主表先取同一轮所有 rank 的最大值，再计算均值和 P50，单位为微秒。`tail50_p50` 只统计末 50 轮；采样不足 50 轮时使用全部样本。`rank0_p50` 单独统计 rank 0，便于核对参考仓的输出口径。原始样本全部保留，不自动剔除异常值。

前 3／10／20 轮均值在样本不足时使用已有样本。`report.txt` 在运行中标记为 `RUNNING`，完成精度检查、采样和释放后才标记为 `PASS`。若途中报错，已完成的 case 仍保存在文件中；此时还需提供报错机器的首错回显。

这里保留 C++ 用例的随机路由和均匀权重。参考仓的 Python 用例另有均衡路由及权重设置，因此比较两仓时还需核对输入和统计字段。先用 `--sync both` 判断相同 kernel、相同输入下的同步影响。

正常运行只打印初始化阶段、每个 BS 的开始和结果，采样循环不逐轮刷屏。每 15 秒的心跳仍显示当前阶段和已完成轮数，便于区分初始化、预热、采样和释放。`--kernel-timeout` 默认 10 秒，约束一次 stream 完成等待；超时会退出 worker，由 torchrun 终止同组进程。该等待也包含尚未完成的同步任务，不能仅凭超时报错判断 Combine 卡死。

汇总中的 `SYNC` 行说明 `host`、`device` 和 `hccl` 对应的实现。JSON 和 CSV 同时保存 `sync_impl`，用于区分这次 SHMEM 同步与此前的 HCCL 同步数据。

显式使用 `--sync hccl` 时，初始化会分别打印建组、张量分配、首次 `all_reduce` 和 stream 同步阶段。脚本还启用 `PYTHONFAULTHANDLER`，进程收到段错误信号时尽量将 Python 调用栈写入 `node-N/console.log`。

Host 检查覆盖了启动参数、采样汇总、单次初始化和入队顺序。CANN 编译及双机 NPU 测量需要在目标环境执行。

## Python 核内打点及 Perfetto

默认 `--profile 2 --profile-iters 10`，每个打点阶段先预热 10 次；干净性能仍测 100 次，两类结果在表格中分别标为 clean/profile。`--sync hccl` 下两类采用同一个 HCCL gate、相同工作流和输入，HCCL 在 SHMEM 之前初始化。profile 缓冲清零在 gate 前，下载/CSV 写入在 end Event 完成后，跨 rank 收集在整组采样结束后使用 Gloo。不要用 profile Event 代替 clean 性能。

每个 BS 都导出：

- `results/ll-python/$RUN_ID/profiles/hccl/<phase>-<variant>/rank-N-profile.csv`：该机器全部本地 rank、全部核、全部 profile 样本。
- 同目录 `diagnostic.csv`、`producers.csv`：第一台保存全局代表核及真实 Dispatch 回传来源。
- 同目录 `perfetto/bs32-p50-i*.perfetto.json`、`bs32-worst-i*.perfetto.json`：第一台汇集所有 rank 和全部核，选择同一个真实全局 P50/最慢轮次。每个 BS 都有对应文件。
- `profiling-hccl.txt`：第一台自动生成并打屏的紧凑分析文本，包含每 BS 的 clean/profile mean、P50、min/max 和核内计数，沿用 C++ 的 D/S/R/G/T 解释及 62 KiB 分块。

使用其他同步时，路径中的 hccl 对应改为 host/device。CSV 保留 ticks，Perfetto 默认按 1 GHz 换算（可传 `--profile-frequency-hz`，0 只落 CSV）；每条核轨道以自身 entry 为原点，不能据此相减跨卡时钟、推算 RDMA 单程时延或启动偏差。P50 图选排序中间的一次真实迭代，不拼接不同轮次。

新增 producer 导出使用独立 coalesce 构建选项，不再把 ready 模式复用的身份位误认成合包，避免错误扣减 DIRECT 核数或将 RDMA 标为 packed/relay。算子数据路径不受此诊断修复影响。

回显独立于 Perfetto：每轮每 rank 除最长发送/接收核、最晚 Put、合包核外，另保留
最晚准备、最晚准备 drain、最长通知、最长 group-ready 窗口、最大 M4 取数等待
的核，去重后汇总。`X` 行报告同一真实全局轮次的准备/通知/group-ready/输入等待
极值及 rank/core；`m4` 固定列出本地/机内读取发起、等待、计算发起、输出等待、
打包、输入 drain 累计 ticks，不依赖 top4 是否把它们选中。RDMA 暂存/发布/CQ、
最终缺失 flag 的 token/slot/block、producer 链路和重复拖尾频次仍保留。
独立阶段最大值与累计计数不能相加当成一条跨 rank 流水；回显也不把 put 返回
当作对端数据已到达。所有新增筛选发生在 Host 采样后，无新增 Device 热路径打点。
