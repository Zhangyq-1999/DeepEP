"""Exercise the real changing-payload check: both banks must reject old data."""
import os
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_changed_payload_rejects_stale_banks_and_restores_input(tmp_path):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG")
    main = (HERE / "main.cpp").read_text(encoding="utf-8")
    methods = main[main.index("    void flipExpertSign("):main.index("    template<class Peers>")]
    source = tmp_path / "reuse.cpp"
    source.write_text(r'''
#include <cassert>
#include <cstring>
#include <stdexcept>
#include "options.hpp"
enum {ACL_MEMCPY_DEVICE_TO_HOST,ACL_MEMCPY_HOST_TO_DEVICE};
int aclrtMemcpy(void* dst,size_t capacity,const void* src,size_t bytes,int) {
    assert(bytes<=capacity); std::memcpy(dst,src,bytes); return 0;
}
void check(int status,const char*) {assert(!status);}
struct Output {
    std::vector<uint16_t> data;
    void download(void* dst,size_t n) {assert(n==data.size()*2);std::memcpy(dst,data.data(),n);}
};
struct Case {
    struct {uint8_t *expandXOut,*sendCountsOut;} dispatch;
    size_t expertBytes;
    std::vector<uint16_t> input;
    Output* output;
''' + methods + r'''
};
int main() {
    for(bool bf16:{false,true}) {
        llbench::Options o; o.experts=4;o.hidden=3584;
        std::vector<uint16_t> original(o.hidden*2);
        for(unsigned i=0;i<original.size();++i) original[i]=llbench::encode(0.5f+(i%4)*0.125f,bf16);
        auto expert=original;
        int counts[4]={0,0,0,2};
        Output out{original};
        Case c{{reinterpret_cast<uint8_t*>(expert.data()),reinterpret_cast<uint8_t*>(counts)},
               expert.size()*2,original,&out};
        // Model alternating banks, both initially contain +x. The test checks
        // the checker, not hardware cache coherency or the Combine algorithm.
        std::vector<uint16_t> bank[2]={original,original};
        for(unsigned g=0;g<4;++g) {
            if(g%2==0)c.flipExpertSign(o);
            out.data=bank[g%2];
            bool rejected=false;
            try {c.precision(bf16);}catch(const std::runtime_error&){rejected=true;}
            assert(rejected); // every old bank value must be distinguishable
            bank[g%2]=expert;out.data=expert;
            assert(c.precision(bf16)==0);
        }
        assert(expert==original && c.input==original);
        for(int invalid:{-1,3}) {
            counts[3]=invalid;bool rejected=false;
            try {c.flipExpertSign(o);}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && c.input==original && expert==original);
        }
        counts[3]=0;c.flipExpertSign(o);c.flipExpertSign(o);
        assert(c.input==original && expert==original);
    }
}
''', encoding="utf-8")
    exe = tmp_path / "reuse.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O1", "-I"+str(HERE),
                    str(source), "-o", str(exe)], check=True, timeout=120)
    subprocess.run([str(exe)], check=True, timeout=20)
