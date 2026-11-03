#include "simt_test_stub.h"
#include "kernels/combine/low_latency/combine_simt.h"
#include "producer_map.hpp"
#include <cassert>
#include <algorithm>
#include <random>
using namespace CombineDedup;
int main()
{
    std::mt19937 random(7919);
    // Actual VFs vs independent serial selection, including non-divisible
    // chunks, all ownership offsets and multi-chunk accumulated output.
    for (unsigned n : {1U, 31U, 32U, 33U, 127U, 256U, 511U, 512U}) {
        std::vector<int32_t> table(n*3), types(n);
        for (unsigned i=0;i<n;++i) { types[i]=1+random()%3; table[i*3]=random()%32; }
        for (unsigned core=0;core<28;++core) for (unsigned owner : {0U,7U,27U}) {
            uint32_t work[64]{}, summary[8]{};
            std::vector<int32_t> actual(n+8,-1), expected;
            uint32_t ordinal=0;
            for (unsigned i=0;i<n;++i) if (types[i]==2 && table[i*3]>=8 && table[i*3]<16) {
                if ((owner+ordinal)%28==core) expected.push_back(1000+i);
                ++ordinal;
            }
            asc_vf_call<PartialSelectCountVf>(dim3(32),table.data(),types.data(),work,n,8U,16U,32U);
            asc_vf_call<PartialSelectWriteVf>(dim3(32),work,summary,actual.data(),n,1000U,owner,core,28U,3U);
            assert(summary[0]==ordinal && summary[1]==0);
            assert(std::equal(expected.begin(),expected.end(),actual.begin()+3));
            assert(actual[2]==-1 && actual[3+expected.size()]==-1);
        }
        types[0]=2; table[0]=-1;
        uint32_t w[64]{}, s[8]{}; int32_t rows[512]{};
        asc_vf_call<PartialSelectCountVf>(dim3(32),table.data(),types.data(),w,n,0U,4U,32U);
        asc_vf_call<PartialSelectWriteVf>(dim3(32),w,s,rows,n,0U,0U,0U,28U,0U);
        assert(s[1]);
    }
    // Inactive slots, every possible missing block, K=32 and poison flags.
    for (unsigned k : {1U,2U,8U,31U,32U}) for (unsigned blocks : {1U,15U,31U,85U}) {
        std::vector<float> flags(k*blocks*8,0);
        uint32_t work[64]{}, summary[8]{};
        const uint32_t active=k==32?0xffffffffU:(1U<<k)-1U;
        auto probe=[&](uint32_t mask) {
            asc_vf_call<ResultFlagsVf>(dim3(32),flags.data(),work,k,blocks,mask);
            asc_vf_call<ResultFlagsSummaryVf>(dim3(32),work,summary,k,blocks);
        };
        for(unsigned i=0;i<k*blocks;++i)flags[i*8]=1;
        probe(active); assert(summary[0]==0 && summary[2]==0);
        for(unsigned i=0;i<k*blocks;++i) {
            flags[i*8]=0; probe(active);
            assert(summary[0]==(1U<<(i/blocks)) && summary[1]==i && summary[2]==1);
            probe(active & ~(1U<<(i/blocks))); assert(summary[0]==0);
            flags[i*8]=1;
        }
        flags[0]=2; probe(active); assert(summary[2]==1); // exact marker check
    }
    // Producer mapping includes both final transport paths and stable cores.
    std::vector<int32_t> types{1,2,3,1,2}, table{5,0,0, 6,1,1, 7,2,2, 0,3,3, 1,4,4};
    auto map=llbench::return_producers(types,table,3,8,8,8,4,4,56,28,false);
    unsigned found=0;
    for(auto v:map)if(v){++found;assert((v&0xffffU)==4);assert(((v>>24)&255)==1 || ((v>>24)&255)==2);}
    assert(found==4);
    bool rejected=false; types[2]=1;table[6]=-1;
    try {llbench::return_producers(types,table,3,8,8,8,4,4,56,28,false);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}
