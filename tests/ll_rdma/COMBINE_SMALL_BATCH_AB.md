# 低时延 Combine 小批量优化对照

## 优化范围

分支 `perf/ll-combine-small-batch` 保留原版 Combine，新增可选的优化路径。编译开关 `DEEPEP_LL_COMBINE_SMALL_BATCH` 默认为 `OFF`；设为 `ON` 后重新编译安装，才会启用优化。

这轮优化面向 BS 不超过 72 的场景，主要减少任务准备和空闲缓冲槽的处理开销。

| 位置 | 原版 | 优化版 |
| --- | --- | --- |
| FIRST_HOP 任务筛选 | 每处理一个源组，重新从 GM 读取类型表和路由表 | 表能放入 UB 时，筛选阶段只读取一次，按原顺序生成本核任务列表 |
| 等待组内数据就绪 | 先等待，再筛选 FIRST_HOP 任务 | 先准备任务列表，再执行原有等待；数据到齐后开始 Get 和归约 |
| TERMINAL 暂存、DIRECT 回传 | 初始化并回收全部缓冲槽的事件 | 只处理本核实际会使用的缓冲槽；DIRECT 空任务核跳过发送准备 |

路由快照复用归约开始前空闲的 3 个 FP32 计算缓冲区。任务列表使用原有区域，输入、输出和通知缓冲区保持原布局。表超过可用空间时，优化版自动使用原扫表流程。

任务列表保存行号。执行具体任务时，算子仍按原流程读取对应的路由条目和成员表。

两版保留相同的任务归属、处理顺序、Get 汇集、部分归约、回传和源端最终归约。权重计算、浮点累加顺序、数据包格式及完成通知均沿用原实现。Dispatch、SHMEM 和 RDMA 队列配置保持相同。

本分支沿用原拓扑：物理主机包含 16 卡，EP16 使用 G4 逻辑去重组，EP32 及以上使用 G16。单机 EP16 通过 MTE 通信，跨物理主机的回传仍走 RDMA。本次对比保留这套分组规则。

## 单机 16 卡 A/B/A

在独立的 Conda 环境中操作。以下命令假定已经加载 CANN 环境，且当前仓库已按 `dependencies.lock.json` 编译 SHMEM。

```bash
git fetch origin
git switch perf/ll-combine-small-batch
git pull --ff-only
source tests/ll_rdma/env.sh

export DEEPEP_LL_STAGE_TIMER=OFF
export DEEPEP_LL_REFERENCE_PROFILE=OFF
export DEEPEP_LL_COUNT_HISTOGRAM=OFF
export DEEPEP_LL_METADATA_CORES=4
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_TYPE=Release

# 设置为本机用于通信初始化的 IP。
: "${MASTER_ADDR:?请先设置 MASTER_ADDR}"

set -euo pipefail
run_variant() {
    local variant="$1" run_name="$2"
    if [[ "$variant" == optimized ]]; then
        export DEEPEP_LL_COMBINE_SMALL_BATCH=ON
    else
        export DEEPEP_LL_COMBINE_SMALL_BATCH=OFF
    fi
    python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
    python tests/ll_stage_timer/run.py \
        --backend rdma --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --num-processes 16 --num-nodes 1 --node-rank 0 \
        --master-addr "$MASTER_ADDR" --master-port 29671 --shmem-port 29672 \
        --num-tokens 1 2 4 8 16 32 48 64 72 --num-max-tokens-per-rank 256 \
        --hidden 3584 --num-topk 8 --num-experts 384 --dtype bf16 \
        --do-cpu-sync 0 --group-backend hccl --use-1g-huge-pages 0 \
        --route normal --seed 0 --warmup 10 --num-iters 100 --perf-tail 50 \
        --correctness-repeats 3 --repeat-combine 3 --num-cpu-timeout-secs 300 \
        --combine-variant "$variant" --output-dir "results/$run_name"
}

run_variant baseline combine-a1
run_variant optimized combine-b
run_variant baseline combine-a2
```

这里的 `rdma` 表示使用原分支的混合通信后端；单机 16 卡之间仍走 MTE。`--combine-variant` 核对已安装的实现，结果文件也会记录变体。构建缓存按编译开关区分两版。

用例先检查精度，再分别采集 Dispatch 和 Combine。三轮对照固定形状、路由种子、预热和采样参数，沿用原有采样流程。第一次正式采样前执行原有 Exp 预备计算，采样循环内保持连续调用。

## 结果判断

查看各结果目录的 `summary.csv`，筛选 `operation=combine`。先确认精度检查通过，以及 `summary.json` 中的路由哈希和采样配置一致，再比较时延。

- `mean_kernel_us`：所有 rank 在统计窗口内的平均 kernel 时延。
- `max_rank_mean_kernel_us`：最慢 rank 的平均 kernel 时延。
- `p99_kernel_us`：统计窗口内的 p99 时延。
- `ranks.csv` 中的 `min_us`、`max_us`：各 rank 的最小和最大时延。

本例采集 100 轮，汇总后 50 轮。各 rank 的 profiler 目录保留 `samples.json`，可用于检查前后轮次和异常值。A1 与 A2 的差异可以辅助判断环境波动；B 的变化需要同时参考平均值、范围和拖尾 rank。

本地 CPU 检查已覆盖新旧任务列表一致性、空任务、尾部行、倾斜路由、UB 容量回退和延迟搬运下的事件回收。CANN 编译、设备精度和性能仍待验证，目前没有实测加速结论。
