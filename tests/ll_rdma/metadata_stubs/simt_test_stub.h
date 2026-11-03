#pragma once
// Sequential lanes are sufficient: production VFs communicate only across
// separate calls, separated by PIPE_V. This does not model device timing.
#define __simt_vf__
#define __launch_bounds__(n)
struct dim3 { unsigned x; explicit dim3(unsigned n):x(n) {} };
inline thread_local struct { unsigned x; } threadIdx;
template<auto F, class... Args> void asc_vf_call(dim3 block, Args... args)
{
    for (unsigned lane = 0; lane < block.x; ++lane) { threadIdx.x = lane; F(args...); }
}
