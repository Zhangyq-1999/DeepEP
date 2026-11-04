#pragma once
#include "combine_coalesce_layout.h"
#include "combine_profile_device.h"
#include "ep_memory_server_dedup_buffer_layout.h"
#include "../../dispatch/low_latency/device/ll_transport.h"

namespace CombineDedup {
using namespace AscendC;
// Compile-time experiment only. Producers retain their mathematical ownership;
// four former DIRECT cores compact returns and unpack peer frames concurrently
// with the original final reduction. No device-wide or cross-rank barrier.
class CoalescedReturn {
    friend struct CoalescedReturnTestAccess;
    using Args=DispatchDedup::CommArgs;
    CombineCoalesce::Layout l_{};
    const __gm__ Args* args_=nullptr;
    CombineProfile::View p_;
    uint64_t base_=0, window_=0;
    uint32_t epoch_=0, count_=0;
    static constexpr bool server_=DEEPEP_LL_COMBINE_COALESCE==2;
    static constexpr bool split_=DEEPEP_LL_COMBINE_COALESCE_ROWS==1;
    __aicore__ inline GlobalTensor<uint8_t> Local(uint64_t off) {
        return LlTransport::Tensor(args_->peerMems[args_->rank]+base_+off,true);
    }
    __aicore__ inline void Load(LocalTensor<uint8_t> ub,uint64_t off,uint32_t bytes) {
        DataCopy(ub,Local(off),bytes); LlTransport::Sync<HardEvent::MTE2_S>(7U);
    }
    __aicore__ inline void Store(uint64_t off,LocalTensor<uint8_t> ub,uint32_t bytes) {
        LlTransport::Sync<HardEvent::S_MTE3>(7U);
        DataCopy(Local(off),ub,bytes); LlTransport::Sync<HardEvent::MTE3_S>(7U);
    }
    __aicore__ inline void Header(LocalTensor<uint8_t> ub,uint32_t rows,uint32_t total,uint32_t chunk,
                                 uint32_t peer,uint32_t magic) {
        auto x=ub.ReinterpretCast<uint32_t>();
        x.SetValue(0,magic); x.SetValue(1,l_.packet); x.SetValue(2,rows); x.SetValue(3,total);
        x.SetValue(4,chunk); x.SetValue(5,uint32_t(args_->rank)); x.SetValue(6,peer); x.SetValue(7,epoch_);
    }
    __aicore__ inline void WaitProducers(LocalTensor<uint8_t> ub) {
        const auto start=p_.DetailClock();
        auto flags=ub.ReinterpretCast<float>();
        auto offsets=ub[3072U].ReinterpretCast<uint32_t>();
        auto values=ub[3584U].ReinterpretCast<float>();
        auto mask=ub[4096U];
        auto selected=ub[4352U].ReinterpretCast<uint32_t>();
        auto indices=offsets.ReinterpretCast<int32_t>();
        CreateVecIndex(indices, int32_t(0), 128U);
        PipeBarrier<PIPE_V>(); Muls(indices,indices,int32_t(32),128U);
        PipeBarrier<PIPE_V>(); Adds(indices,indices,int32_t(28),128U);
        LlTransport::Sync<HardEvent::V_S>(7U);
        while (true) {
            LlTransport::Sync<HardEvent::V_MTE2>(7U);
            DataCopy(ub,Local(4096U),l_.cores*32U);
            LlTransport::Sync<HardEvent::MTE2_V>(7U);
            Gather(values,flags,offsets,0U,l_.cores);
            LlTransport::Sync<HardEvent::V_S>(7U);
            for(uint32_t c=l_.direct;c<l_.cores-l_.recv;++c) values.SetValue(c,float(epoch_));
            LlTransport::Sync<HardEvent::S_V>(7U);
            Compares(mask,values,float(epoch_),CMPMODE::EQ,l_.cores);
            PipeBarrier<PIPE_V>();
            uint64_t matches=0;
            GatherMask(selected,offsets,mask.ReinterpretCast<uint32_t>(),true,l_.cores,{1U,1U,0U,0U},matches);
            LlTransport::Sync<HardEvent::V_S>(7U);
            if(matches==l_.cores) break;
        }
        // Counts are consumed from a new snapshot after the last-word epoch
        // checks. Do not reuse a snapshot racing the count publication.
        LlTransport::Sync<HardEvent::V_MTE2>(7U);
        DataCopy(ub,Local(4096U),l_.cores*32U);
        LlTransport::Sync<HardEvent::MTE2_S>(7U);
        p_.Elapsed(CombineProfile::CoalesceWaitTicks,start);
        p_.Mark(CombineProfile::CoalesceReady);
    }
    __aicore__ inline void PackAndSend(uint32_t bucket,LocalTensor<uint8_t> ub) {
        const uint32_t peer=CombineCoalesce::Target(l_,uint32_t(args_->rank),bucket,server_);
        if(LlTransport::SameHost(args_,peer)) return;
        const auto begin=p_.DetailClock();
        // WaitProducers left a stable 32B count record per core in ub[0:2304].
        auto counts=ub.ReinterpretCast<uint32_t>();
        auto record=ub[8192U];
        uint32_t rows=0;
        const uint64_t dst=l_.send+uint64_t(bucket)*l_.frameCapacity;
        for(uint32_t core=0;core<l_.cores;++core) {
            if(CombineCoalesce::Worker(l_,core)) continue;
            const uint32_t n=counts.GetValue(core*8U+1U);
            if(n>l_.coreRows) { Trap(); return; }
            for(uint32_t i=0;i<n;++i) {
                const uint64_t from=CombineCoalesce::Row(l_,core,i);
                Load(record,from,32U);
                const uint32_t target=record.ReinterpretCast<uint32_t>().GetValue(0U);
                if(CombineCoalesce::Bucket(l_,target,server_)!=bucket) continue;
                if(rows>=l_.bucketRows) { Trap(); return; }
                Load(record,from,l_.record);
                const uint64_t to=dst+CombineCoalesce::FrameOffset(l_,rows,split_)+CombineCoalesce::kHead+
                    (split_?0U:uint64_t(rows)*l_.record);
                DataCopy(Local(to),record,l_.record);
                LlTransport::Sync<HardEvent::MTE3_MTE2>(7U);
                ++rows;
            }
        }
        p_.Elapsed(CombineProfile::CoalescePackTicks,begin);
        p_.Add(CombineProfile::CoalesceRows,rows);
        auto control=ub[6144U];  // independent of counts and payload
        const uint32_t chunks=split_&&rows?rows:1U;
        for(uint32_t i=0;i<chunks;++i) {
            const uint32_t n=split_&&rows?1U:rows;
            const uint64_t off=CombineCoalesce::FrameOffset(l_,i,split_);
            const uint32_t bytes=CombineCoalesce::FrameBytes(l_,n);
            Header(control,n,rows,i,peer,CombineCoalesce::kHeaderMagic);
            Store(dst+off,control,32U);
            Header(control,n,rows,i,peer,CombineCoalesce::kTailMagic);
            Store(dst+off+bytes-32U,control,32U);
            const auto put=p_.DetailClock();
            p_.MarkOnce(CombineProfile::RdmaFirstPut);
            LlTransport::Put(args_,peer,LlTransport::RemoteAddress(args_,peer,base_+l_.inbox+
                uint64_t(args_->rank)*l_.frameCapacity+off),
                args_->peerMems[args_->rank]+base_+dst+off,bytes,control);
            p_.Mark(CombineProfile::RdmaLastPut);
            p_.Elapsed(CombineProfile::RdmaPutTicks,put);
            p_.Add(CombineProfile::RdmaPackets); p_.Add(CombineProfile::CoalesceBytes,bytes);
            // Reclaim queue credits without reusing any registered payload.
            if((i+1U)%16U==0U || i+1U==chunks) {
                const auto quiet=p_.DetailClock();
                LlTransport::Quiet(peer,control);
                p_.Elapsed(CombineProfile::RdmaQuietTicks,quiet); p_.Add(CombineProfile::RdmaQuietCalls);
            }
        }
    }
    __aicore__ inline void Receive(uint32_t source,LocalTensor<uint8_t> ub) {
        if(LlTransport::SameHost(args_,source) || (server_ && source%l_.host!=uint32_t(args_->rank)%l_.host)) return;
        auto control=ub[6144U]; auto header=control.ReinterpretCast<uint32_t>(); auto record=ub[8192U];
        const uint64_t inbox=l_.inbox+uint64_t(source)*l_.frameCapacity;
        uint32_t total=0, chunks=1U;
        for(uint32_t i=0;i<chunks;++i) {
            const uint64_t off=CombineCoalesce::FrameOffset(l_,i,split_);
            const auto wait=p_.DetailClock();
            // Header is provisional until the matching tail is visible. Never
            // trust a count to form a DMA address without bounds checks.
            do { Load(control,inbox+off,32U); }
            while(header.GetValue(0)!=CombineCoalesce::kHeaderMagic || header.GetValue(7)!=epoch_);
            Load(control,inbox+off,32U);
            const uint32_t n=header.GetValue(2), observed=header.GetValue(3);
            if(observed>l_.bucketRows || n>l_.bucketRows || header.GetValue(4)!=i ||
               header.GetValue(5)!=source || header.GetValue(6)!=uint32_t(args_->rank) || header.GetValue(1)!=l_.packet ||
               n!=(split_&&observed?1U:observed) || (i && observed!=total)) { Trap(); return; }
            total=observed; chunks=split_&&total?total:1U;
            const uint32_t bytes=CombineCoalesce::FrameBytes(l_,n);
            do { Load(control,inbox+off+bytes-32U,32U); }
            while(header.GetValue(0)!=CombineCoalesce::kTailMagic || header.GetValue(7)!=epoch_ ||
                  header.GetValue(2)!=n || header.GetValue(3)!=total || header.GetValue(4)!=i);
            p_.Elapsed(CombineProfile::CoalesceReceiveWaitTicks,wait);
            p_.MarkOnce(CombineProfile::CoalesceFirstReady);
            p_.Mark(CombineProfile::CoalesceLastReady);
            const auto unpack=p_.DetailClock();
            for(uint32_t j=0;j<n;++j) {
                Load(record,inbox+off+CombineCoalesce::kHead+uint64_t(j)*l_.record,l_.record);
                const auto meta=record.ReinterpretCast<uint32_t>();
                const uint32_t peer=meta.GetValue(0), slot=meta.GetValue(1);
                if(peer>=l_.world || slot>=l_.bs*l_.k || !LlTransport::SameHost(args_,peer) ||
                   (!server_ && peer!=uint32_t(args_->rank))) { Trap(); return; }
                // Preserve every 512B data-with-flag block on the final MTE
                // hop. Final reduction keeps its existing all-block check.
                const uint64_t to=window_+COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET+uint64_t(slot)*l_.packet;
                DataCopy(LlTransport::Tensor(args_->peerMems[peer]+to,true),record[32U],l_.packet);
                LlTransport::Sync<HardEvent::MTE3_MTE2>(7U);
                p_.Add(CombineProfile::CoalesceUnpackRows);
                if(peer!=uint32_t(args_->rank)) p_.Add(CombineProfile::CoalesceForwardRows);
            }
            p_.Elapsed(CombineProfile::CoalesceUnpackTicks,unpack);
        }
    }
public:
    __aicore__ inline void SetCombineProfile(CombineProfile::DeviceProfile* profile) { p_.Bind(profile); }
    __aicore__ inline void Configure(const __gm__ Args* args,uint32_t bs,uint32_t h,uint32_t k,uint32_t experts,
                                     uint32_t recv,uint64_t window,LocalTensor<uint8_t> scratch) {
        args_=args; window_=window; base_=window+CombineCoalesce::kBase;
        if(!LlTransport::Rdma(args) || !CombineCoalesce::Make(uint32_t(args->rankSize),args->reserved0,bs,h,k,experts,
            GetBlockNum(),recv,server_,l_)) { Trap(); return; }
        LlTransport::Check(args,uint32_t(args->rank),base_,l_.bytes);
        Load(scratch,GetBlockIdx()*32U,32U);
        auto state=scratch.ReinterpretCast<uint32_t>(); epoch_=state.GetValue(0)+1U;
        if(!epoch_ || epoch_>CombineCoalesce::kMaxEpoch) { Trap(); return; }
        state.SetValue(0,epoch_); Store(GetBlockIdx()*32U,scratch,32U);
    }
    __aicore__ inline void Init(const __gm__ Args*,uint32_t bytes) { if(bytes!=l_.packet) Trap(); }
    __aicore__ inline void Enqueue(uint32_t peer,uint64_t offset,LocalTensor<uint8_t> packet,bool=false) {
        const uint64_t results=window_+COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET;
        if(count_>=l_.coreRows || offset<results || (offset-results)%l_.packet ||
           (offset-results)/l_.packet>=l_.bs*l_.k || peer>=l_.world) { Trap(); return; }
        const auto start=p_.DetailClock();
        const uint64_t row=CombineCoalesce::Row(l_,GetBlockIdx(),count_++);
        DataCopy(Local(row+32U),packet,l_.packet);
        LlTransport::Sync<HardEvent::MTE3_S>(7U);
        // Like the original SDK scratch use, the retired packet body may now
        // be overwritten; its 512B flag positions remain untouched.
        auto meta=packet.ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<8U;++i) meta.SetValue(i,0U);
        meta.SetValue(0,peer); meta.SetValue(1,uint32_t((offset-results)/l_.packet));
        Store(row,packet,32U);
        p_.Elapsed(CombineProfile::CoalesceStageTicks,start);
    }
    __aicore__ inline void ProducerDone(LocalTensor<uint8_t> scratch) {
        PipeBarrier<PIPE_ALL>();
        auto state=scratch.ReinterpretCast<uint32_t>();
        for(uint32_t i=0;i<8U;++i) state.SetValue(i,0U);
        scratch.ReinterpretCast<float>().SetValue(7,float(epoch_)); state.SetValue(1,count_);
        Store(4096U+GetBlockIdx()*32U,scratch,32U);
        p_.Mark(CombineProfile::CoalesceProducerDone);
    }
    __aicore__ inline bool IsWorker() const { return CombineCoalesce::Worker(l_,GetBlockIdx()); }
    __aicore__ inline uint32_t DirectCores() const { return l_.direct; }
    __aicore__ inline void Run(LocalTensor<uint8_t> scratch) {
        PipeBarrier<PIPE_ALL>(); p_.Set(CombineProfile::CoalesceWorker,1U);
        const uint32_t worker=GetBlockIdx()-l_.direct;
        WaitProducers(scratch);
        p_.Mark(CombineProfile::CoalesceSendBegin);
        for(uint32_t b=worker;b<l_.buckets;b+=CombineCoalesce::kWorkers) PackAndSend(b,scratch);
        p_.Mark(CombineProfile::CoalesceSendEnd);
        for(uint32_t r=worker;r<l_.world;r+=CombineCoalesce::kWorkers) Receive(r,scratch);
        p_.Mark(CombineProfile::CoalesceReceiveEnd);
    }
    __aicore__ inline void SubmitPending() {}
    __aicore__ inline bool Pending() const { return false; }
    __aicore__ inline bool HasStaged() const { return false; }
    __aicore__ inline void Flush(LocalTensor<uint8_t>) {}
};
}
