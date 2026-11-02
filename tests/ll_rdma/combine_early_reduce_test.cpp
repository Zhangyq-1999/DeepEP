// CPU execution of production M5 with delayed contributions and M4 stores.
#define DEEPEP_TEST_DEFERRED_MTE3
#include "metadata_stubs/kernel_operator.h"
#include <cassert>
#include <algorithm>
#include <random>
#include <type_traits>
using GM_ADDR = uint8_t*;
using half = _Float16;
struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() = default;
    explicit bfloat16_t(float value) {
        uint32_t word; std::memcpy(&word, &value, 4);
        word += 0x7fffU + ((word >> 16U) & 1U); bits = uint16_t(word >> 16U);
    }
    operator float() const { uint32_t word = uint32_t(bits) << 16U; float value; std::memcpy(&value, &word, 4); return value; }
};
namespace AscendC {
enum class RoundMode { CAST_NONE, CAST_RINT };
template<int> uint32_t GetBitCount(uint64_t bits) { return uint32_t(__builtin_popcountll(bits)); }
template<class D, class S> void Cast(LocalTensor<D> d, LocalTensor<S> s, RoundMode, uint32_t n) {
    for (uint32_t i=0;i<n;++i) d.p[i] = D(float(s.p[i]));
}
inline uint32_t contributions=0;
template<class T> void Muls(LocalTensor<T> d, LocalTensor<T> s, T weight, uint32_t n) {
    ++contributions; for(uint32_t i=0;i<n;++i)d.p[i]=s.p[i]*weight;
}
template<class T> void Add(LocalTensor<T> d, LocalTensor<T> a, LocalTensor<T> b, uint32_t n) {
    for(uint32_t i=0;i<n;++i)d.p[i]=a.p[i]+b.p[i];
}
inline void ReduceSum(LocalTensor<float> d,LocalTensor<float> s,LocalTensor<float>,uint32_t mask,uint32_t n,uint32_t stride) {
    assert(mask==1 && stride==1);float total=0;for(uint32_t i=0;i<n;++i)total+=s.p[i*8];d.p[0]=total;
}
inline uint8_t* packets=nullptr;
inline uint32_t packetBytes=0,blocks=0,tick=0,earlyObserved=0;
inline std::vector<uint32_t> arrival;
inline std::vector<bool> published;
inline void PublishReady() {
    ++tick;assert(tick<100000);
    for(uint32_t c=0;c<arrival.size();++c)if(!published[c] && arrival[c] && arrival[c]<=tick){
        for(uint32_t b=0;b<blocks;++b){
            float flag=1;std::memcpy(packets+c*packetBytes+b*512+480,&flag,4);
        }
        published[c]=true;
    }
    if(contributions && std::find(published.begin(),published.end(),false)!=published.end())earlyObserved=1;
}
template<class T> void TestDataCopy(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyParams params) {
    if constexpr(std::is_same_v<T,float>) PublishReady();
    else if(packets && reinterpret_cast<uint8_t*>(src.p)>=packets &&
            reinterpret_cast<uint8_t*>(src.p)<packets+arrival.size()*packetBytes){
        auto index=(reinterpret_cast<uint8_t*>(src.p)-packets)/packetBytes;
        assert(published[index]); // no data read ahead of all block flags
    }
    DataCopy(dst,src,params);
}
template<class T> void TestDataCopyPad(LocalTensor<T> d,GlobalTensor<T> s,DataCopyExtParams p,DataCopyPadExtParams<T> pad) {
    DataCopyPad(d,s,p,pad);
}
template<class T> void TestDataCopyPad(GlobalTensor<T> d,LocalTensor<T> s,DataCopyExtParams p) {
    for(uint32_t i=0;i<p.count;++i)
        deferredMte3.push_back({reinterpret_cast<uint8_t*>(d.p)+i*(p.bytes+p.dstStride),
            reinterpret_cast<uint8_t*>(s.p)+i*((p.bytes+31)/32*32+p.srcStride*32),p.bytes,++issuedMte3});
}
}
#define DataCopy TestDataCopy
#define DataCopyPad TestDataCopyPad
#include "kernels/combine/low_latency/combine_server_dedup_source_aggregate.h"
#undef DataCopy
#undef DataCopyPad
using namespace AscendC;
using namespace CombineDedup;

template<class T> void Check(uint32_t bs,uint32_t h,uint32_t k,uint32_t participant,bool dedup,uint32_t delay,uint32_t partialTasks) {
    const uint32_t recv=8;
    auto layout=EpCombineM45Layout(100,bs,h,k,recv);
    assert(layout.inputLanes && layout.totalBytes<=kEpServerDedupCombineRecvUbBytes);
    const uint32_t usable=kEpServerDedupCombineRecvUbBytes-kEpServerDedupCombineResidentUbBytes;
    std::vector<uint64_t> arena((usable+128)/8,0xababababababababULL);
    auto scratch=reinterpret_cast<uint8_t*>(arena.data())+64;
    std::vector<uint32_t> masks(bs);
    std::vector<float> weights(bs*k);
    blocks=(h*2+479)/480;packetBytes=blocks*512;
    std::vector<uint64_t> result(uint64_t(bs)*k*packetBytes/8,0);
    packets=reinterpret_cast<uint8_t*>(result.data());
    arrival.assign(bs*k,0);published.assign(bs*k,true);
    for(uint32_t t=0;t<bs;++t){
        masks[t]=(1U<<(t%k))|(1U<<(k-1));
        if(t%3==0)masks[t]=k==32?0xffffffffU:(1U<<k)-1;
        for(uint32_t j=0;j<k;++j){
            weights[t*k+j]=float((t+j)%7+1)/8;
            bool active=!dedup || ((masks[t]>>j)&1U);
            arrival[t*k+j]=active?(delay?1+(j*37+t*19)%151:1):0;published[t*k+j]=!active;
            for(uint32_t x=0;x<h;++x){
                T value=T(float(int((t*13+j*7+x)%97)-48)/32);
                uint32_t byte=x*2;
                std::memcpy(packets+(t*k+j)*packetBytes+byte/480*512+byte%480,&value,2);
            }
        }
    }
    std::vector<uint16_t> output(bs*h+64,0x6a6a);
    std::vector<std::vector<uint8_t>> oldStores(layout.outputLanes,std::vector<uint8_t>(layout.outputBytes));
    // Simulate M4 output DMA still reading each physical output slot.
    if(dedup)for(uint32_t lane=0;lane<layout.outputLanes;++lane){
        auto src=scratch+layout.outputOffset+lane*layout.outputBytes;
        std::memset(src,31+lane,layout.outputBytes);
        deferredMte3.push_back({oldStores[lane].data(),src,layout.outputBytes,++issuedMte3});
        SetFlag<HardEvent::MTE3_V>(lane);
    }
    ServerDedupSourceAggregate op;
    const uint8_t dtype=std::is_same_v<T,half>?DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16:DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if(dedup)op.Init({scratch},packets,reinterpret_cast<GM_ADDR>(masks.data()),reinterpret_cast<GM_ADDR>(weights.data()),
        reinterpret_cast<GM_ADDR>(output.data()),bs,h,k,dtype,participant,recv,layout,partialTasks,true,true);
    else op.InitDirectShared({scratch},packets,reinterpret_cast<GM_ADDR>(weights.data()),reinterpret_cast<GM_ADDR>(output.data()),
        bs,h,k,dtype,participant,recv,layout,true);
    tick=contributions=earlyObserved=0;
    uint32_t retired=op.Process();
    if(dedup){for(uint32_t lane=0;lane<layout.outputLanes;++lane)if(!((retired>>lane)&1U))WaitFlag<HardEvent::MTE3_V>(lane);}
    else op.DrainM6();
    SetFlag<HardEvent::MTE3_S>(7);WaitFlag<HardEvent::MTE3_S>(7);
    for(auto& direction:events)for(bool live:direction)assert(!live);
    assert(deferredMte3.empty());
    const uint32_t count=bs/recv+(participant<bs%recv),begin=participant*(bs/recv)+std::min(participant,bs%recv);
    for(uint32_t t=0;t<bs;++t)for(uint32_t x=0;x<h;++x){
        if(t<begin || t>=begin+count){assert(output[t*h+x]==0x6a6a);continue;}
        float sum=0;bool first=true;
        for(uint32_t j=0;j<k;++j)if(!dedup || ((masks[t]>>j)&1U)){
            T value;uint32_t byte=x*2;std::memcpy(&value,packets+(t*k+j)*packetBytes+byte/480*512+byte%480,2);
            float product=float(value)*weights[t*k+j];sum=first?product:sum+product;first=false;
        }
        T expected=T(sum);uint16_t bits;std::memcpy(&bits,&expected,2);assert(output[t*h+x]==bits);
    }
    for(uint32_t t=begin;t<begin+count;++t)for(uint32_t j=0;j<k;++j)for(uint32_t b=0;b<blocks;++b){
        uint32_t flag;std::memcpy(&flag,packets+(t*k+j)*packetBytes+b*512+480,4);assert(flag==0);
    }
    for(auto& store:oldStores)if(dedup){uint8_t expected=31+(&store-&oldStores[0]);for(auto byte:store)assert(byte==expected);}
    for(uint32_t i=0;i<8;++i){assert(arena[i]==0xababababababababULL);assert(arena[arena.size()-1-i]==0xababababababababULL);}
    for(uint32_t i=bs*h;i<output.size();++i)assert(output[i]==0x6a6a);
}
int main(){
    for(uint32_t bs:{1,8,9,16,32,72,73})for(uint32_t k:{1,6,8,32})for(bool dedup:{false,true})
        for(uint32_t participant:{0,7})for(uint32_t delay:{0,1}){
            Check<half>(bs,256,k,participant,dedup,delay,5);
            Check<bfloat16_t>(bs,256,k,participant,dedup,delay,0);
        }
    for(uint32_t h:{3584,7168})for(uint32_t partial:{0,1,7,11}){
        Check<half>(72,h,8,0,true,1,partial);Check<bfloat16_t>(72,h,6,7,false,1,0);
    }
    std::puts("PASS production M5: delayed arrivals, BF16/FP16 results, M4 retirement, masks, tails and guards");
}
