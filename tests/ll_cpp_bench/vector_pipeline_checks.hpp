#pragma once
// Check dependencies between Vector instructions in the real Aggregate body.
// Arithmetic executes on CPU, but any RAW/WAR/WAW across distinct instructions
// without a PIPE_V barrier fails. Exact aliasing within one Muls/Add is legal.
#include <vector>
#include <cassert>
#include <cstdint>
namespace VectorChecks {
struct Access { uintptr_t begin, end; bool write; };
inline std::vector<Access> pending;
inline unsigned barriers = 0, multiplies = 0;
template<class T> Access Range(AscendC::LocalTensor<T> t, unsigned n, bool write) {
    return {reinterpret_cast<uintptr_t>(t.p), reinterpret_cast<uintptr_t>(t.p + n), write};
}
inline void Issue(std::initializer_list<Access> current) {
    for (const auto& a : current) for (const auto& b : pending)
        assert(!(a.begin < b.end && b.begin < a.end && (a.write || b.write)));
    pending.insert(pending.end(), current.begin(), current.end());
}
template<int Pipe> void Barrier() {
    static_assert(Pipe == PIPE_V);
    pending.clear(); ++barriers;
}
template<class D, class S>
void Cast(AscendC::LocalTensor<D> d, AscendC::LocalTensor<S> s, AscendC::RoundMode mode, unsigned n) {
    Issue({Range(d,n,true), Range(s,n,false)}); AscendC::Cast(d,s,mode,n);
}
template<class T> void Muls(AscendC::LocalTensor<T> d, AscendC::LocalTensor<T> s, T w, unsigned n) {
    Issue({Range(d,n,true), Range(s,n,false)}); ++multiplies; AscendC::Muls(d,s,w,n);
}
template<class T>
void Add(AscendC::LocalTensor<T> d, AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, unsigned n) {
    Issue({Range(d,n,true), Range(a,n,false), Range(b,n,false)}); AscendC::Add(d,a,b,n);
}
}
