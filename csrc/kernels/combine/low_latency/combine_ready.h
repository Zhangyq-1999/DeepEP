#pragma once
#include "kernel_operator.h"
#include "combine_profile_device.h"
#include "combine_poll.h"
#include "../../dispatch/low_latency/device/ll_transport.h"
#include "ep_memory_server_dedup_buffer_layout.h"

#include "combine_ready_layout.h"

namespace CombineDedup {
#if DEEPEP_LL_COMBINE_READY > 0
// One reader per receive core. Mode 1 consumes the original private ready row.
// Mode 2 shares a rank-ready row, with a generation and a read-consumed ACK.
// A publisher cannot overwrite that bank's flag until EVERY receiver saw it.
template <uint32_t Group>
class CombineGroupReady {
public:
    __aicore__ inline void Init(const __gm__ DispatchDedup::CommArgs* args, uint64_t control, uint32_t core,
                                uint32_t send, uint32_t recv, AscendC::LocalTensor<float> input,
                                AscendC::LocalTensor<float> work, AscendC::LocalTensor<float> indices,
                                AscendC::LocalTensor<float> result, AscendC::LocalTensor<float> publication,
                                CombineProfile::View profile)
    {
        args_ = args;
        control_ = control;
        core_ = core;
        send_ = send;
        recv_ = recv;
        observed_ = 0U;
        input_ = input.ReinterpretCast<uint32_t>();
        work_ = work.ReinterpretCast<int32_t>();
        indices_ = indices.ReinterpretCast<uint32_t>();
        result_ = result.ReinterpretCast<uint8_t>();
        publication_ = publication.ReinterpretCast<uint32_t>();
        profile_ = profile;
        if (core < send) return;
        const uint64_t start = profile_.DetailClock();
        AscendC::CreateVecIndex(indices_.ReinterpretCast<int32_t>(), int32_t(0), 64U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(indices_.ReinterpretCast<int32_t>(), indices_.ReinterpretCast<int32_t>(), int32_t(32), 64U);
        Sync<AscendC::HardEvent::V_S>();
#if DEEPEP_LL_COMBINE_READY == 2
        // Each core owns its counter in each existing window bank. No shared
        // scalar increment, and no change to Dispatch's 0/1 window rotation.
        auto epoch = Tensor(uint32_t(args_->rank), kSharedEpoch + core * 32U);
        AscendC::DataCopyPad(input_, epoch, {1U, 32U, 0U, 0U, 0U}, {false, 0U, 0U, uint32_t(0)});
        Sync<AscendC::HardEvent::MTE2_S>();
        epoch_ = uint32_t(input_.GetValue(0U)) + 1U;
        AscendC::Duplicate(publication_, epoch_, 8U);
        Sync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(epoch, publication_, {1U, 32U, 0U, 0U, 0U});
        Sync<AscendC::HardEvent::MTE3_S>();
#endif
        profile_.Elapsed(CombineProfile::ReadySetupTicks, start);
    }
    __aicore__ inline bool All() const
    {
        return observed_ == ((1U << Group) - 1U);
    }
    __aicore__ inline uint32_t Poll()
    {
        if (All()) return observed_;
        profile_.MarkOnce(CombineProfile::ReadyBegin);
        const uint64_t start = profile_.DetailClock();
#if DEEPEP_LL_COMBINE_READY == 2
        const uint64_t offset = kSharedReady;
        const uint32_t expected = epoch_;
#else
        const uint64_t offset = COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET + (core_ - send_) * 512U;
        const uint32_t expected = 0x3f800000U;
#endif
        observed_ |= uint32_t(ReadMask(offset, Group, expected));
        profile_.Add(CombineProfile::ReadyPollCalls);
        profile_.Add(CombineProfile::ReadyPollMisses, All() ? 0U : 1U);
        profile_.Elapsed(CombineProfile::ReadySnapshotTicks, start);
        profile_.MaxElapsed(CombineProfile::ReadyPollMaxTicks, start);
        if (All()) profile_.Mark(CombineProfile::ReadyObserved);
        return observed_;
    }
    __aicore__ inline void Wait()
    {
        while (!All()) {
            Poll();
            if (!All()) PauseAfterFlagMiss(profile_);
        }
    }
    __aicore__ inline void Retire()
    {
        profile_.Mark(CombineProfile::ReadyRetireBegin);
        Wait();
#if DEEPEP_LL_COMBINE_READY == 2
        Write(uint32_t(args_->rank), kSharedConsumed + (core_ - send_) * 32U);
#else
        // Includes delayed, unused peers. Never clear before their write arrived.
        AscendC::Duplicate(indices_, uint32_t(0), 128U);
        Sync<AscendC::HardEvent::V_MTE3>();
        auto dest = Tensor(uint32_t(args_->rank),
                           COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET + (core_ - send_) * 512U);
        AscendC::DataCopyPad(dest, indices_, {1U, 512U, 0U, 0U, 0U});
#endif
        profile_.Mark(CombineProfile::ReadyEnd);
        profile_.Mark(CombineProfile::ReadyRetireEnd);
    }
    __aicore__ inline void Publish(uint32_t peer)
    {
        const uint64_t begin = profile_.DetailClock();
        // ACK is written into this source rank's local memory by the receiver.
        // Initial ACK/counter are both zero; uint32 wrap compares exact bits.
        while (ReadMask(kSharedAck + (peer % Group) * 32U, 1U, epoch_ - 1U) != 1U) PauseAfterFlagMiss(profile_);
        profile_.Elapsed(CombineProfile::ReadyAckWaitTicks, begin);
        Write(peer, kSharedReady + (uint32_t(args_->rank) % Group) * 32U);
        profile_.Add(CombineProfile::NotifyStores);
    }
    __aicore__ inline void Release()
    {
        if (core_ != send_) return;
        const uint64_t begin = profile_.DetailClock();
        while (ReadMask(kSharedConsumed, recv_, epoch_) != ((uint64_t(1) << recv_) - 1U)) PauseAfterFlagMiss(profile_);
        profile_.Elapsed(CombineProfile::ReadyConsumedWaitTicks, begin);
        const uint32_t base = uint32_t(args_->rank) / Group * Group;
        for (uint32_t peer = base; peer < base + Group; ++peer)
            Write(peer, kSharedAck + (uint32_t(args_->rank) % Group) * 32U);
    }

private:
    template <AscendC::HardEvent E>
    __aicore__ inline void Sync()
    {
        // 7 is separate from the live input/output lane tickets (0..6).
        AscendC::SetFlag<E>(7U);
        AscendC::WaitFlag<E>(7U);
    }
    __aicore__ inline AscendC::GlobalTensor<uint32_t> Tensor(uint32_t peer, uint64_t offset)
    {
        if (!LlTransport::SameHost(args_, peer) || !args_->peerMems[peer]) {
            AscendC::Trap();
        }
        AscendC::GlobalTensor<uint32_t> t;
        t.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(args_->peerMems[peer] + control_ + offset));
        t.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        return t;
    }
    __aicore__ inline uint64_t ReadMask(uint64_t offset, uint32_t count, uint32_t expected)
    {
        Sync<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyPad(input_, Tensor(uint32_t(args_->rank), offset), {1U, count * 32U, 0U, 0U, 0U},
                             {false, 0U, 0U, uint32_t(0)});
        Sync<AscendC::HardEvent::MTE2_V>();
        AscendC::Duplicate(work_, int32_t(~expected), 64U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(work_, input_.ReinterpretCast<int32_t>(), indices_, 0U, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compares(result_, work_, int32_t(expected), AscendC::CMPMODE::EQ, 64U);
        Sync<AscendC::HardEvent::V_S>();
        return result_.ReinterpretCast<uint64_t>().GetValue(0U) & ((uint64_t(1) << count) - 1U);
    }
    __aicore__ inline void Write(uint32_t peer, uint64_t offset)
    {
        // Publication source is resident and immutable until kernel PIPE_ALL.
        AscendC::DataCopyPad(Tensor(peer, offset), publication_, {1U, 32U, 0U, 0U, 0U});
    }
    const __gm__ DispatchDedup::CommArgs* args_ = nullptr;
    AscendC::LocalTensor<int32_t> work_;
    AscendC::LocalTensor<uint32_t> input_;
    AscendC::LocalTensor<uint32_t> indices_, publication_;
    AscendC::LocalTensor<uint8_t> result_;
    uint64_t control_ = 0;
    uint32_t core_ = 0, send_ = 0, recv_ = 0, epoch_ = 0, observed_ = 0;
    CombineProfile::View profile_;
};
#endif
}  // namespace CombineDedup
