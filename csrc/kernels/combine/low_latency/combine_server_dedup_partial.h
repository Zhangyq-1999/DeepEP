// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#ifndef COMBINE_DEDUP_EP_SERVER_DEDUP_PARTIAL_H
#define COMBINE_DEDUP_EP_SERVER_DEDUP_PARTIAL_H

#include "kernel_operator.h"
#include "combine_ready.h"
#if DEEPEP_LL_COMBINE_READY == 1
#include "combine_ready_select.h"
#endif
#include "combine_profile_device.h"
#include "../../dispatch/low_latency/device/ll_transport.h"
#include "combine_rdma_return.h"
#include "combine_stage_wire.h"
#include "../../dispatch/low_latency/common/comm_args.h"
#include "../../dispatch/low_latency/common/dispatch_dedup_types.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "ep_memory_server_dedup_combine_config.h"
#include "combine_server_dedup_select.h"
#ifdef DEEPEP_LL_COMBINE_SIMT_META
#include "combine_simt.h"
#endif
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
#include "combine_partial_plan.h"
#endif

namespace CombineDedup {
using namespace AscendC;

// 中文审计：M4按目标server筛选元数据、聚合贡献并回传partial结果。
// 输出地址与MTE3完成事件交给M5继续使用；退出本类不表示所有远端消费已经完成。
// Recv-core M4 scans type/routing metadata per destination server. Its output
// addresses and completion events remain live through the M5 handoff.
template<uint32_t Group>
class ServerDedupPartial {
public:
    __aicore__ inline void SetProfile(CombineProfile::DeviceProfile* p)
    {
        profile_.Bind(p);
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
        plan_.SetProfile(p);
#endif
    }
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, TPipe* pipe, LocalTensor<uint8_t> scratch,
                                GM_ADDR expert, GM_ADDR table2, GM_ADDR table3, GM_ADDR types, uint32_t tokens,
                                uint32_t bs, uint32_t h, uint32_t k, uint32_t experts, uint8_t dtype, uint64_t window,
                                const EpCombineM45WorkspaceLayout& layout, ReturnBatch* batch)
    {
        args_ = args;
        rdmaBatch_ = batch;

        recvCores_ = EpCombineRecvCoreCount(GetBlockNum());
        pipe_ = pipe;
        scratch_ = scratch[layout.phaseOffset];
        tokens_ = tokens;
        bs_ = bs;
        h_ = h;
        k_ = k;
        localExperts_ = experts / uint32_t(args->rankSize);
        dtype_ = dtype;
        window_ = window;
        blocks_ = (h * 2U + 479U) / 480U;
        packetBytes_ = blocks_ * 512U;
        inputBytes_ = layout.inputBytes;
        stageStride_ = packetBytes_;
        stageBase_ = window + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET + uint64_t(bs) * k * packetBytes_;
        inputLanes_ = layout.inputLanes;
        outputLanes_ = layout.outputLanes;
        if (inputLanes_ == 0U || outputLanes_ == 0U) {
            Trap();
            return;
        }
        rows_ = Allocate<int32_t>(EpCombineM4TaskBytes(tokens, recvCores_));
#if DEEPEP_LL_COMBINE_READY == 1
        const uint32_t bytes=EpCombineReadyTaskBytes(tokens,recvCores_);
        readyDeps_=Allocate<int32_t>(bytes);
        readyWork_=Allocate<int32_t>(bytes);
        readyTasks_=Allocate<int32_t>(bytes);
        readyMask_=Allocate<uint8_t>(EpCombineAlign((bytes/4U+7U)/8U,32U));
#endif
        scan_ = scratch_[used_];
        scanRows_ = EpCombineM4ScanRows(h, k);
        table2Ub_ = Allocate<int32_t>(EpCombineAlign(kEpServerDedupCombineM4TableRows * 12U, 32U));
        table3Ub_ = Allocate<uint32_t>(EpCombineAlign(kEpServerDedupCombineM4TableRows * k * 8U, 32U));
#ifdef DEEPEP_LL_COMBINE_SIMT_META
        table2Cache_ = table2Ub_;
        table3Cache_ = table3Ub_;
#endif
        cast_ = scratch.ReinterpretCast<float>();
        product_ = scratch[layout.computeBytes].ReinterpretCast<float>();
        sum_ = scratch[2U * layout.computeBytes].ReinterpretCast<float>();
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
        planScratch_ = scratch;
        planScratchBytes_ = layout.phaseOffset;
#endif
        low_ = Allocate<uint16_t>(blocks_ * 480U);
        inputs_ = scratch[layout.inputOffset].ReinterpretCast<uint16_t>();
        outputs_ = scratch[layout.outputOffset];
        expert_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(expert));
        table2_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table2));
        table3_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(table3));
        types_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(types));
    }

#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
    __aicore__ inline void PrepareTasks()
    {
        planned_ = plan_.Prepare(planScratch_, planScratchBytes_, rows_, types_, table2_, tokens_,
                                 uint32_t(args_->rankSize), uint32_t(args_->rank),
                                 GetBlockIdx() - (GetBlockNum() - recvCores_), recvCores_);
    }
#endif

#if DEEPEP_LL_COMBINE_READY == 1
    __aicore__ inline void SetReady(CombineGroupReady<Group>* ready) { ready_=ready; }
#endif
    __aicore__ inline uint32_t Process()
    {
        if (LlTransport::Rdma(args_)) rdmaBatch_->Init(args_, packetBytes_);
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) SetFlag<HardEvent::V_MTE2>(lane);
        uint64_t flagMask[2] = {0xffU, 0U};
        for (uint32_t lane = 0; lane < outputLanes_; ++lane) {
            Duplicate<float>(outputs_[lane * packetBytes_].ReinterpretCast<float>()[120U], 1.0f, flagMask,
                             uint8_t(blocks_), 1U, 16U);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t servers = uint32_t(args_->rankSize) / Group;
        const uint32_t ownServer = uint32_t(args_->rank) / Group;
        uint32_t owner = 0U, completed = 0U;
#if DEEPEP_LL_COMBINE_READY == 1
        completed=ProcessReadyTasks(servers,ownServer);
#else
        for (uint32_t step = 1U; step <= servers; ++step) {
#ifdef DEEPEP_LL_COMBINE_SIMT_META
            cacheEnd_ = 0U;  // server scan aliases both metadata caches
#endif
            const uint64_t scanStart = profile_.DetailClock();
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
            const uint32_t begin = planned_ && step > 1U ? plan_.End(step - 2U) : 0U;
            const uint32_t end = planned_ ? plan_.End(step - 1U)
                : SelectServer(CombineReturnServer<Group>(servers, ownServer, step - 1U), owner, step == 1U);
            const uint32_t taskEnd = end;
            profile_.Elapsed(CombineProfile::PartialScanTicks, scanStart);
            for (uint32_t task = begin; task < end; ++task) {
#else
            const uint32_t count = SelectServer(CombineReturnServer<Group>(servers, ownServer, step - 1U), owner, step == 1U);
            const uint32_t taskEnd = count;
            profile_.Elapsed(CombineProfile::PartialScanTicks, scanStart);
            for (uint32_t task = 0; task < count; ++task) {
#endif
                const uint32_t row = uint32_t(rows_.GetValue(task));
                const uint64_t metaStart = profile_.DetailClock();
#ifdef DEEPEP_LL_COMBINE_SIMT_META
                LoadTaskBatch(task, taskEnd);
#else
                LoadTask(row);
#endif
                profile_.Elapsed(CombineProfile::PartialMetadataTicks, metaStart);
#ifdef DEEPEP_LL_COMBINE_EARLY_ISSUE
                // Issue before the next packet queues its Vector -> MTE3 wait.
                // The previous packet owns a different UB output lane; the
                // batch waits for its GM staging before using that UB as scratch.
                const uint64_t issueStart = profile_.DetailClock();
                rdmaBatch_->SubmitPending();
                profile_.Elapsed(CombineProfile::EarlyIssueTicks, issueStart);
#endif
                const uint32_t source = uint32_t(table2Ub_.GetValue(0U));
                const uint32_t token = uint32_t(table2Ub_.GetValue(1U));
                const uint32_t topk = uint32_t(table2Ub_.GetValue(2U));
                const uint32_t members = table3Ub_.GetValue(0U);  // full uint32 N, not uint16
                if (members == 0U || members >= k_) {
                    Trap();
                    return completed;
                }
                if (dtype_ == DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16)
                    Aggregate<half>(row, source, members + 1U);
                else
                    Aggregate<bfloat16_t>(row, source, members + 1U);
                // Publish a group's last packet immediately, including the
                // one-task case. Only defer when another packet will follow.
                Send(completed % outputLanes_, source, token, topk,
                     CombineDeferReturn(outputLanes_) && task + 1U < taskEnd);
                ++completed;
            }
            // Reuse phase scratch with its Vector dependency, not an MTE3 drain.
        }
#endif
        const uint64_t drainStart = profile_.DetailClock();
        for (uint32_t lane = 0; lane < inputLanes_; ++lane) WaitFlag<HardEvent::V_MTE2>(lane);
        profile_.Elapsed(CombineProfile::PartialInputDrainTicks, drainStart);
        // Post the final staged packet before M5 can wait on peer returns.
        // Isolated GM slots survive M5's UB reuse; tail CQ retirement is owned
        // by the enclosing kernel, after M5, not by this temporary object.
        rdmaBatch_->SubmitPending();
        // Output events and their UB sources remain owned by the shared ring.
        // The caller hands off only MTE2/Vector before M5 reuses phase metadata.
        return completed;
    }

private:
#if DEEPEP_LL_COMBINE_READY == 1
    __aicore__ inline uint32_t ProcessReadyTasks(uint32_t servers,uint32_t ownServer)
    {
        profile_.Mark(CombineProfile::PlanBegin);
        const uint64_t start=profile_.DetailClock();
        uint32_t owner=0U,count=0U;
        // Keep the baseline per-core ownership and server order. Build once;
        // blocked tasks retain their identity rather than rescanning the table.
        for(uint32_t step=0;step<servers;++step)
            count=SelectServer(CombineReturnServer<Group>(servers,ownServer,step),owner,step==0U,count);
        const uint32_t rounded=EpCombineAlign(count,64U);
        Duplicate(readyDeps_,int32_t(0x10000U),rounded);
        Sync<HardEvent::V_S>();
        for(uint32_t task=0;task<count;++task) {
            LoadTask(uint32_t(rows_.GetValue(task)));
            const uint32_t n=table3Ub_.GetValue(0U);
            if(n==0U || n>=k_) {Trap();return 0U;}
            uint32_t mask=0U;
            for(uint32_t member=1;member<=n;++member) {
                const uint32_t rank=(table3Ub_.GetValue(member*2U)&0xffffU)/localExperts_;
                if(rank/Group!=uint32_t(args_->rank)/Group) {Trap();return 0U;}
                mask|=1U<<(rank%Group);
            }
            const uint32_t source=uint32_t(table2Ub_.GetValue(0U));
            readyDeps_.SetValue(task,int32_t(mask|(LlTransport::RemoteWrite(args_,source)?0U:kReadyTaskLocal)));
        }
        profile_.Elapsed(CombineProfile::ReadyPlanTicks,start);
        profile_.Mark(CombineProfile::PlanEnd);
        uint32_t completed=0U;
        while(completed<count) {
            const uint32_t observed=ready_->Poll();
            const uint64_t filterStart=profile_.DetailClock();
            const uint32_t missing=((1U<<Group)-1U)^observed;
            uint32_t runnable=SelectReadyCombineTasks(readyDeps_,readyWork_,readyTasks_,readyMask_,count,missing,true);
            if(!runnable) runnable=SelectReadyCombineTasks(readyDeps_,readyWork_,readyTasks_,readyMask_,count,missing,false);
            profile_.Elapsed(CombineProfile::ReadyFilterTicks,filterStart);
            profile_.Add(CombineProfile::ReadyTaskRounds);
            if(!runnable) {
                profile_.Add(CombineProfile::ReadyTaskStalls);
                PauseAfterFlagMiss(profile_);
                continue;
            }
            profile_.MarkOnce(CombineProfile::ReadyFirstTask);
            // An independent token can run before a blocked token; member order
            // inside Aggregate remains unchanged. Never read an unready stage.
            for(uint32_t index=0;index<uint32_t(runnable);++index) {
                const uint32_t task=uint32_t(readyTasks_.GetValue(index));
                const uint32_t row=uint32_t(rows_.GetValue(task));
                LoadTask(row);
                const uint32_t source=uint32_t(table2Ub_.GetValue(0U));
                const uint32_t token=uint32_t(table2Ub_.GetValue(1U));
                const uint32_t topk=uint32_t(table2Ub_.GetValue(2U));
                const uint32_t n=table3Ub_.GetValue(0U);
                if(dtype_==DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16) Aggregate<half>(row,source,n+1U);
                else Aggregate<bfloat16_t>(row,source,n+1U);
                Send(completed%outputLanes_,source,token,topk,false);
                readyDeps_.SetValue(task,int32_t(0x10000U));
                ++completed;
                if(!ready_->All()) {
                    profile_.Add(CombineProfile::ReadyTasksBeforeAll);
                    profile_.Add(CombineProfile::ReadyRemoteBeforeAll,LlTransport::RemoteWrite(args_,source)?1U:0U);
                }
            }
        }
        profile_.Mark(CombineProfile::ReadyTasksDone);
        return completed;
    }
#endif
    template <class T>
    __aicore__ inline LocalTensor<T> Allocate(uint32_t bytes)
    {
        auto tensor = scratch_[used_].template ReinterpretCast<T>();
        used_ += bytes;
        return tensor;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const TEventID id = pipe_->FetchEventID(E);
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }
    __aicore__ inline uint32_t SelectServer(uint32_t server, uint32_t& owner, bool validateRanks, uint32_t append = 0U)
    {
        uint32_t count = append;
        const uint32_t localCore = GetBlockIdx() - (GetBlockNum() - recvCores_);
        const uint32_t vectorBytes = EpCombineAlign(scanRows_ * 4U, 256U);
        const uint32_t maskBytes = EpCombineAlign((scanRows_ + 7U) / 8U, 32U);
        auto table = scan_.ReinterpretCast<int32_t>();
        auto ranks = scan_[scanRows_ * 12U].ReinterpretCast<int32_t>();
        auto indices = scan_[scanRows_ * 12U + vectorBytes].ReinterpretCast<int32_t>();
        auto typeMask = scan_[scanRows_ * 12U + 2U * vectorBytes];
        auto mask = typeMask[maskBytes], upper = mask[maskBytes];
        for (uint32_t begin = 0U; begin < tokens_; begin += scanRows_) {
            const uint32_t span = tokens_ - begin < scanRows_ ? tokens_ - begin : scanRows_;
            // The scan aliases low_ and table caches. Retire their Vector users
            // before MTE2 overwrites them; output slots remain independent/live.
            // Input lanes 0..2 keep seeded V_MTE2 events live. Use a distinct
            // event in this direction for phase scratch, never FetchEventID(0).
            const uint64_t loadStart = profile_.DetailClock();
            SetFlag<HardEvent::V_MTE2>(7U);
            WaitFlag<HardEvent::V_MTE2>(7U);
            DataCopyPad(ranks, types_[begin], {1U, span * 4U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            DataCopyPad(table, table2_[uint64_t(begin) * 3U], {1U, span * 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
            SetFlag<HardEvent::MTE2_V>(7U);
            WaitFlag<HardEvent::MTE2_V>(7U);
            profile_.Elapsed(CombineProfile::PartialScanLoadTicks, loadStart);
            const uint64_t selectStart = profile_.DetailClock();
#ifdef DEEPEP_LL_COMBINE_SIMT_META
            // 256 B counts/bits and 32 B summary reuse the original scan space.
            auto* work = reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr());
            auto* summary = reinterpret_cast<__ubuf__ uint32_t*>(typeMask.GetPhyAddr());
            asc_vf_call<PartialSelectCountVf>(dim3(32),
                reinterpret_cast<__ubuf__ int32_t*>(table.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(ranks.GetPhyAddr()), work,
                span, server * Group, (server + 1U) * Group, uint32_t(args_->rankSize));
            PipeBarrier<PIPE_V>();
            asc_vf_call<PartialSelectWriteVf>(dim3(32), work, summary,
                reinterpret_cast<__ubuf__ int32_t*>(rows_.GetPhyAddr()),
                span, begin, owner, localCore, recvCores_, count);
            Sync<HardEvent::V_S>();
            auto result = typeMask.ReinterpretCast<uint32_t>();
            if (validateRanks && result.GetValue(1U)) { Trap(); return 0U; }
            const uint32_t selected = result.GetValue(0U);
            const uint32_t first = ServerDedupFirstOwned(localCore, owner, recvCores_);
            if (first < selected) count += (selected - 1U - first) / recvCores_ + 1U;
            owner = (owner + selected) % recvCores_;
#else
            Compares(typeMask, ranks, int32_t(2), CMPMODE::EQ, (span + 63U) / 64U * 64U);
            CreateVecIndex(indices, int32_t(0), span);
            PipeBarrier<PIPE_V>();
            Muls(indices, indices, int32_t(12), span);
            PipeBarrier<PIPE_V>();
            Gather(ranks, table, indices.ReinterpretCast<uint32_t>(), 0U, span);
            PipeBarrier<PIPE_V>();
            CreateVecIndex(indices, int32_t(begin), span);
            PipeBarrier<PIPE_V>();
            if (validateRanks) {
                uint64_t typed = 0U, valid = 0U;
                GatherMask(table, indices, typeMask.ReinterpretCast<uint32_t>(), true, span, {1U, 1U, 0U, 0U}, typed);
                ServerDedupRankMask(mask, upper, ranks, 0, int32_t(args_->rankSize), span);
                And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                    typeMask.ReinterpretCast<uint16_t>(), int32_t((span + 15U) / 16U));
                PipeBarrier<PIPE_V>();
                GatherMask(table, indices, mask.ReinterpretCast<uint32_t>(), true, span, {1U, 1U, 0U, 0U}, valid);
                Sync<HardEvent::V_S>();
                if (valid != typed) {
                    Trap();
                    return 0U;
                }
            }
            ServerDedupRankMask(mask, upper, ranks, int32_t(server * Group), int32_t((server + 1U) * Group), span);
            And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                typeMask.ReinterpretCast<uint16_t>(), int32_t((span + 15U) / 16U));
            PipeBarrier<PIPE_V>();
            uint64_t selected = 0U;
            GatherMask(table, indices, mask.ReinterpretCast<uint32_t>(), true, span, {1U, 1U, 0U, 0U}, selected);
            Sync<HardEvent::V_S>();
            const uint32_t first = ServerDedupFirstOwned(localCore, owner, recvCores_);
            owner = (owner + uint32_t(selected)) % recvCores_;
            for (uint32_t task = first; task < uint32_t(selected); task += recvCores_)
                rows_.SetValue(count++, table.GetValue(task));
#endif
            profile_.Elapsed(CombineProfile::PartialScanVectorTicks, selectStart);
        }
        return count;
    }
    __aicore__ inline void LoadTask(uint32_t row)
    {
        // Route is reloaded by original row so the list needs only 4B/task.
        // Table3 is exactly one K*8B row, fetched only for an owned type2 task.
        DataCopyPad(table2Ub_, table2_[uint64_t(row) * 3U], {1U, 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
        DataCopyPad(table3Ub_, table3_[uint64_t(row) * k_ * 2U], {1U, k_ * 8U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
        Sync<HardEvent::MTE2_S>();
    }
#ifdef DEEPEP_LL_COMBINE_SIMT_META
    __aicore__ inline void LoadTaskBatch(uint32_t task, uint32_t end)
    {
        if (task >= cacheEnd_) {
            cacheBegin_ = task;
            cacheEnd_ = task + 8U < end ? task + 8U : end;
            // Existing 384 B table2 cache fits eight padded 32 B rows. Table3
            // already reserves 32 rows; no input/output lane is taken away.
            for (uint32_t i = task; i < cacheEnd_; ++i) {
                const uint32_t row = uint32_t(rows_.GetValue(i)), slot = i - task;
                DataCopyPad(table2Cache_[slot * 8U], table2_[uint64_t(row) * 3U],
                    {1U, 12U, 0U, 0U, 0U}, {false, 0U, 0U, 0});
                DataCopyPad(table3Cache_[slot * EpCombineAlign(k_ * 8U, 32U) / 4U], table3_[uint64_t(row) * k_ * 2U],
                    {1U, k_ * 8U, 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            }
            Sync<HardEvent::MTE2_S>();
            profile_.Add(CombineProfile::MetadataBatches);
        }
        table2Ub_ = table2Cache_[(task - cacheBegin_) * 8U];
        table3Ub_ = table3Cache_[(task - cacheBegin_) * EpCombineAlign(k_ * 8U, 32U) / 4U];
    }
#endif
    __aicore__ inline void Load(uint32_t row, uint32_t source, uint32_t member)
    {
        const uint32_t lane = (inputSequence_ + member) % inputLanes_;
        const uint64_t reuseStart = profile_.DetailClock();
        WaitFlag<HardEvent::V_MTE2>(lane);
        profile_.Elapsed(CombineProfile::PartialReuseWaitTicks, reuseStart);
        const uint64_t issueStart = profile_.DetailClock();
        GlobalTensor<uint16_t> data;
        if (member == 0U) {
            data = expert_[uint64_t(row) * h_];

            DataCopyPad(inputs_[lane * inputBytes_ / 2U], data, {1U, h_ * 2U, 0U, 0U, 0U},
                        {false, 0U, 0U, uint16_t(0)});
        } else {
            const uint32_t packed = table3Ub_.GetValue(member * 2U);
            const uint32_t expert = packed & 0xffffU, q = packed >> 16U;
            const uint32_t memberRank = expert / localExperts_, localExpert = expert % localExperts_;
            const uint64_t slot =
                (((uint64_t(source) + memberRank) % uint32_t(args_->rankSize)) * localExperts_ + localExpert) * bs_ + q;
            if (memberRank / Group != uint32_t(args_->rank) / Group) { Trap(); return; }

            LoadStageWire(args_, memberRank, stageBase_ + slot * stageStride_,
                inputs_[lane * inputBytes_ / 2U], h_ * 2U);
        }
        SetFlag<HardEvent::MTE2_V>(lane);
        profile_.Elapsed(member == 0U ? CombineProfile::PartialLocalIssueTicks : CombineProfile::PartialGetIssueTicks,
                         issueStart);
        profile_.Add(member == 0U ? CombineProfile::PartialLocalContrib : CombineProfile::PartialGetContrib);
    }
    template <class T>
    __aicore__ inline void Aggregate(uint32_t row, uint32_t source, uint32_t contributions)
    {
        const uint32_t preload = contributions < inputLanes_ ? contributions : inputLanes_;
        for (uint32_t member = 0; member < preload; ++member) Load(row, source, member);
#ifdef DEEPEP_LL_COMBINE_PREFETCH_RETURN
        // Let current MTE2 input reads overlap the previous packet's MTE3 wait
        // and Put submission. The old output lane is still owned by the batch.
        const uint64_t issueStart = profile_.DetailClock();
        rdmaBatch_->SubmitPending();
        profile_.Elapsed(CombineProfile::EarlyIssueTicks, issueStart);
#endif
        const auto weights = table3Ub_.ReinterpretCast<float>();
        for (uint32_t member = 0; member < contributions; ++member) {
            const uint32_t lane = (inputSequence_ + member) % inputLanes_;
            const float weight = weights.GetValue(member * 2U + 1U);
            const uint64_t waitStart = profile_.DetailClock();
            WaitFlag<HardEvent::MTE2_V>(lane);
            profile_.Elapsed(member == 0U ? CombineProfile::PartialLocalWaitTicks : CombineProfile::PartialGetWaitTicks,
                             waitStart);
            const uint64_t castStart = profile_.DetailClock();
            Cast(cast_, inputs_[lane * inputBytes_ / 2U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, h_);
            SetFlag<HardEvent::V_MTE2>(lane);
            profile_.Elapsed(CombineProfile::PartialComputeIssueTicks, castStart);
            // Prefetch may reuse this input after Cast, independent of weighted
            // accumulation and the previous token's outstanding partial send.
            if (member + inputLanes_ < contributions) Load(row, source, member + inputLanes_);
            const uint64_t computeStart = profile_.DetailClock();
            PipeBarrier<PIPE_V>();
            if (member == 0U)
                Muls(sum_, cast_, weight, h_);
            else {
                Muls(product_, cast_, weight, h_);
                PipeBarrier<PIPE_V>();
                Add(sum_, sum_, product_, h_);
            }
            PipeBarrier<PIPE_V>();
            profile_.Elapsed(CombineProfile::PartialComputeIssueTicks, computeStart);
        }
        inputSequence_ += contributions;
        const uint64_t castStart = profile_.DetailClock();
        Cast(low_.template ReinterpretCast<T>(), sum_, RoundMode::CAST_RINT, h_);
        PipeBarrier<PIPE_V>();
        profile_.Elapsed(CombineProfile::PartialComputeIssueTicks, castStart);
    }
    __aicore__ inline void Send(uint32_t lane, uint32_t source, uint32_t token, uint32_t topk, bool defer)
    {
        // Only output-lane reuse waits for MTE3; aggregation entry never does.
        const uint64_t reuseStart = profile_.DetailClock();
        WaitFlag<HardEvent::MTE3_V>(lane);
        profile_.Elapsed(CombineProfile::PartialOutputWaitTicks, reuseStart);
        const uint64_t packStart = profile_.DetailClock();
        auto output = outputs_[lane * packetBytes_];
        auto dst = output.ReinterpretCast<int32_t>();
        auto src = low_.ReinterpretCast<int32_t>();
        Copy(dst, src, uint64_t(64U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
        Copy(dst[64U], src[64U], uint64_t(56U), uint8_t(blocks_), {1U, 1U, 16U, 15U});
        SetFlag<HardEvent::V_MTE3>(lane);
        WaitFlag<HardEvent::V_MTE3>(lane);
        profile_.Elapsed(CombineProfile::PartialPackTicks, packStart);

        // This reduced output has one destination; the next use rewrites its
        // body and never reads the SDK's borrowed first 128 bytes.
        const uint64_t offset = window_ + COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET +
            (uint64_t(token) * k_ + topk) * packetBytes_;
        const uint64_t submitStart = profile_.DetailClock();
        SendReturnPacket(*rdmaBatch_, args_, source, offset, output, packetBytes_, defer);
        profile_.SampleMax(CombineProfile::PartialSubmitTicks, CombineProfile::PartialSubmitMaxTicks,
                           CombineProfile::PartialSubmitMaxPeer, CombineProfile::PartialSubmitMaxToken, submitStart,
                           source, token);
        profile_.Add(CombineProfile::PartialSubmitCalls);
        profile_.Add(CombineProfile::PartialRemoteCalls, LlTransport::RemoteWrite(args_, source) ? 1U : 0U);
        SetFlag<HardEvent::MTE3_V>(lane);
    }

#if DEEPEP_LL_COMBINE_READY == 1
    CombineGroupReady<Group>* ready_=nullptr;
    LocalTensor<int32_t> readyDeps_,readyWork_,readyTasks_;
    LocalTensor<uint8_t> readyMask_;
#endif
    CombineProfile::View profile_;
    uint32_t recvCores_ = 0;
    ReturnBatch* rdmaBatch_ = nullptr;
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    TPipe* pipe_ = nullptr;
    LocalTensor<uint8_t> scratch_, outputs_, scan_;
    LocalTensor<int32_t> rows_, table2Ub_;
    LocalTensor<uint32_t> table3Ub_;
    LocalTensor<float> cast_, product_, sum_;
    LocalTensor<uint16_t> low_, inputs_;
    GlobalTensor<uint16_t> expert_;
    GlobalTensor<int32_t> table2_, types_;
    GlobalTensor<uint32_t> table3_;
    uint32_t used_ = 0, tokens_ = 0, bs_ = 0, h_ = 0, k_ = 0, localExperts_ = 0, dtype_ = 0;
    uint32_t blocks_ = 0, packetBytes_ = 0, inputBytes_ = 0, inputLanes_ = 0, outputLanes_ = 0;
    uint32_t inputSequence_ = 0, scanRows_ = 0;
    uint64_t stageStride_ = 0, stageBase_ = 0, window_ = 0;
#ifdef DEEPEP_LL_COMBINE_SIMT_META
    LocalTensor<int32_t> table2Cache_;
    LocalTensor<uint32_t> table3Cache_;
    uint32_t cacheBegin_ = 0U, cacheEnd_ = 0U;
#endif
#ifdef DEEPEP_LL_COMBINE_SMALL_BATCH
    PartialTaskPlan<Group> plan_;
    LocalTensor<uint8_t> planScratch_;
    uint32_t planScratchBytes_ = 0U;
    bool planned_ = false;
#endif
};
}  // namespace CombineDedup
#endif
