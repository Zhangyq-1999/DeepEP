#pragma once
#include <cstdint>

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define LL_PACK_INLINE __aicore__ inline
#else
#define LL_PACK_INLINE inline
#endif
namespace CombineCoalesce {
// Experimental storage in the unused upper half of each 4 GiB window.
// Host admission also checks Dispatch and Combine stay below this boundary.
constexpr uint64_t kBase = 2ULL << 30;
constexpr uint32_t kWorkers = 4U;
constexpr uint32_t kHead = 32U, kTail = 32U;
constexpr uint32_t kHeaderMagic = 0x434f4131U, kTailMagic = 0x434f4132U;
constexpr uint32_t kMaxEpoch = 1048575U;  // exact FP32 comparison of done epochs
struct Layout {
    uint32_t world=0, host=0, bs=0, k=0, cores=0, recv=0, direct=0;
    uint32_t packet=0, record=0, coreRows=0, bucketRows=0, buckets=0;
    uint64_t stage=8192U, send=0, inbox=0, frameCapacity=0, bytes=0;
};
LL_PACK_INLINE uint64_t Align(uint64_t n) { return (n + 511U) / 512U * 512U; }
LL_PACK_INLINE bool Make(uint32_t world, uint32_t host, uint32_t bs, uint32_t h, uint32_t k,
                         uint32_t experts, uint32_t cores, uint32_t recv, bool server, Layout& l)
{
    l = {};
    if (!host || world <= host || world > 64U || world % host || !bs || bs > 65536U ||
        !h || h > 16384U || !k || k > 32U || experts < world || experts % world ||
        cores > 72U || recv < 8U || recv + kWorkers >= cores) return false;
    l.world=world; l.host=host; l.bs=bs; l.k=k; l.cores=cores; l.recv=recv;
    l.direct=cores-recv-kWorkers;
    l.packet=(h*2U+479U)/480U*512U;
    l.record=kHead+l.packet;
    const uint32_t localExperts=experts/world, copies=k<localExperts?k:localExperts;
    uint64_t rows=uint64_t(world)*bs*copies;
    if (rows>10000U) rows=10000U;
    const uint32_t owners=l.direct<recv?l.direct:recv;
    l.coreRows=uint32_t((rows+owners-1U)/owners);
    l.bucketRows=bs*k*(server?host:1U);
    l.buckets=server?world/host:world;
    // Same capacity in all/row variants; row frames carry their own head/tail.
    l.frameCapacity=Align(uint64_t(l.bucketRows)*(l.record+kHead+kTail));
    l.send=Align(l.stage+uint64_t(cores)*l.coreRows*l.record);
    l.inbox=Align(l.send+uint64_t(l.buckets)*l.frameCapacity);
    l.bytes=Align(l.inbox+uint64_t(world)*l.frameCapacity);
    return l.bytes < (2ULL<<30)-(2ULL<<20) && l.frameCapacity <= 0xffffffffULL;
}
LL_PACK_INLINE bool Worker(const Layout& l, uint32_t c) { return c>=l.direct && c<l.cores-l.recv; }
LL_PACK_INLINE uint64_t Row(const Layout& l,uint32_t core,uint32_t row) {
    return l.stage+(uint64_t(core)*l.coreRows+row)*l.record;
}
LL_PACK_INLINE uint32_t Bucket(const Layout& l,uint32_t peer,bool server) { return server?peer/l.host:peer; }
LL_PACK_INLINE uint32_t Target(const Layout& l,uint32_t rank,uint32_t bucket,bool server) {
    return server?bucket*l.host+rank%l.host:bucket;
}
LL_PACK_INLINE uint64_t FrameOffset(const Layout& l,uint32_t row,bool split) {
    return split?uint64_t(row)*(kHead+l.record+kTail):0U;
}
LL_PACK_INLINE uint32_t FrameBytes(const Layout& l,uint32_t rows) { return kHead+rows*l.record+kTail; }
}
#undef LL_PACK_INLINE
