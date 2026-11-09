# ops-transformer Combine 参考基线

`ops-shmem-base` 直接编译 [ubx_reserve 的 Combine 源码](https://gitcode.com/zzg_code/ops-transformer/blob/d1bad1c202ce628c7249bf248f1f967de74ac195/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_roce.h)，固定提交 `d1bad1c202ce628c7249bf248f1f967de74ac195`。构建检查 checkout 版本和已跟踪文件的修改，保留上游许可证。C++ 和 Python 默认都包含该版本，BS 默认 `1,2,4,8,16,24`。

## 保留及适配范围

保留原 kernel 的 scatter、机内 gather、独立跨机发送核、就绪轮询、部分归约、发送批次划分、全核同步及全核最终归约；不加入 DeepEP 的分核、L2、tiny 或其他优化开关。测试配置为非量化 BF16/FP16、H3584/H7168、TopK8、384 专家、开启专家权重和 pre-reduce、不融合 AddRmsNorm。这里不运行 ACLNN Host tiling 或 Dispatch；Host 根据相同路由在计时外构造上游要求的 tiling、expanded 输入、行索引、前缀计数及位图。

干净版源码只替换四处文本类型/头文件表达式：Hcomm include、Hcomm 对象类型、BatchHandle 类型及 ChannelHandle 类型。构建后的全文与固定上游应用这四项替换的结果进行测试比对；其余算法源码保持一致。

通信适配沿用用例的 SHMEM 初始化、对称窗口及每核 QP：每次 `WriteNbi` 调用 SHMEM Put，每个上游 `BatchCommit` 边界做 quiet。**这不等同于 Hcomm 的原生整批 WQE 发布、doorbell 和 CQE 策略**，因此测值代表“上游算法 + SHMEM 传输”，不能宣称复现了原 ACLNN/Hcomm 的绝对时延。没有在 C++ 用例增加 HCCL 初始化或算子调用。Python 可继续选择既有的 HCCL 算子前同步，且其耗时不在 Event 内。

## 拓扑及生命周期

每 server 的参与 rank 数来自 `--ranks-per-host`，不使用物理设备编号计算路由。支持默认 EP8（双机各4卡）、EP32（双机各16卡）、EP64（四机各16卡）以及现有单机形状。机内目标通过 `aclshmem_ptr` 映射；跨机按对端 server 的同一 local rank 回传。保留上游两项一个 server 的通道索引布局。更换物理卡仍用 `--devices`。

共享原有8 GiB对称工作区：前4 GiB为机内窗口，后4 GiB为跨机窗口。上游 kernel 自行推进 epoch/bank，单次调用之间不强制清 flag；版本切换前所有 rank 完成并清理工作区。Python 的参考版也借用同一 session，所有默认版本共用一次 SHMEM 初始化。

预热前检查相同地址上的 `-x,-x,+x,+x` 四代输入，避免把旧数据命中当成有效结果。输入准备、正确性检查、同步及 warmup 都不进入 clean Event。采样逐轮取全局 rank-MAX，保留 mean、min、max、P50/P99 和所有原始样本。

## 编译、回显与验证边界

两台分别运行现有 `build_all.sh` 即可，首次构建会抓取固定上游源码；离线时设置 `OPS_SOURCE_DIR=/path/to/clean/pinned/checkout`。Python 自动安装到 `py311_zyq`，两端需重编，后续 `run_all.sh` 沿用已有 IP 和设备默认值。默认保留原始 `base`、参考版、`input6-l2-route-only` 三个干净对照，增加 `ops-shmem-retire`、`ops-shmem-overlap` 两个实验及其打点版，共7个构建。参考版、route-only 的打点版以及 tiny 改为按需选择。

新实验分别减少批次 quiet、重叠最终规约和发送收尾；仅 BS≤24、EP≤64 生效，详见 [实验与数据](SMALL_BS_AB.md)。以上四处替换及原流程的说明特指 `ops-shmem-base`，不将实验版标成未改动的上游算法。

`ops-shmem-base-profile` 单独编译打点，输出细粒度诊断及 Perfetto；干净参考版的四处适配保持不变。回显 `REF`、summary CSV 和构建/安装目录的 `ops-baseline.txt` 记录固定提交及传输区别。

本地主机检查覆盖小 BS 路由/权重/位图、每机4/16 rank的EP8/32/64映射、批次 quiet、源码适配范围、Python共享初始化及版本选择。没有本地 CANN/NPU，目标设备编译、精度与性能仍须实测；主机模型通过不代表设备验证通过。

## 小 BS 诊断（schema 27）

打点仅进入独立的 `-profile` 构建。构建替换锚点必须唯一；主机测试移除诊断语句后逐字核对基线算法，并检查实验版只改声明和 Process 调度。基线 quiet 边界、包数、epoch、分核、归约和就绪判断不变。

回显 `O` 是实际单核记录，`role=scatter/gather/rdma` 表示前一阶段角色，所有角色之后仍参加最终规约和全核同步；policy2 将同步放到退出前。`active` 是该 rank 全核记录在 Host 侧统计的有效 scatter/gather/RDMA/final 核数，gather 计数需要 profile2。

schema27 保持每核304 words，增加 `policy=0/1/2`（实际生效策略）、`deferred`（省去的批次 quiet 数）、`peers`（收尾 quiet 的实际 peer 数）及 `retire` 起止。Perfetto 显示 `ops deferred transport retirement`；policy2 显示 `ops exit all-core barrier`。累计 quiet ticks 不能当作独立的流水片段，也不能当成网络单程时延。

`oe` 输出初始化、scatter、首个就绪任务、gather 结束、首/末 put、发送核结束、全核同步起止、最终规约初始化/首末就绪/store/结束时间。`op` 输出建表耗时/扫描条数/有效任务数、轮询失败/总次数/耗时、RDMA 包数/实际字节/put 耗时/quiet 耗时/次数/最大值、最终轮询及归约贡献数。

先看首 put 是否受机内 gather 延迟，BS 增长时建表和批次 quiet 是否增大，再看全核同步与最终数据就绪。put、quiet 都是发送端观测，不是网络单程时延；各核入口独立，跨核偏移不能直接相减。终端显示真实 P50/worst 样本的代表核；完整 CSV 和全核 Perfetto 保留。

DeepEP 的逐块尾 flag 诊断改为同 token 首次及每8次重复失败采样一次，实际就绪轮询不变，成功观察仍记录。`sampled=stride/calls/skipped` 标记覆盖率，T 是最后观测到的缺失 flag，不能保证捕获全部瞬时拖尾。新旧 profile 耗时及拖尾频率不能直接混比；性能仍以干净版 mean/P50/min/max 为准。
