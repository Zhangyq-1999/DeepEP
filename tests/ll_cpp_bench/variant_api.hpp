#pragma once
#include "options.hpp"
#include "mpi_coordinator.hpp"
#include "progress.hpp"
#include <acl/acl.h>
#include <cstdlib>

namespace llbench {
constexpr uint64_t workspaceBytes = 8ULL << 30;
// Borrowed resources. Only the runner initializes or releases MPI/ACL/SHMEM.
struct Runtime {
    Options options;
    const Coordinator* peers;
    aclrtStream stream;
    uint8_t* workspace;
    uint8_t* rdmaStage;
    uint64_t capacity, ffts;
    uint32_t cores, rdmaQps;
    const Progress* progress = nullptr;
    void report(const char* stage, unsigned bs = 0, unsigned done = 0, unsigned total = 0) const
    {
        if (progress) progress->report(stage, bs, done, total);
    }
    void synchronize(const char* stage, unsigned bs = 0, unsigned iteration = 0) const
    {
        const int status = aclrtSynchronizeStreamWithTimeout(stream, int32_t(options.kernelTimeout * 1000U));
        if (!status) return;
        std::fprintf(stderr, "FAIL rank=%u device=%u output=%s BS=%u iteration=%u stage=%s status=%d "
                             "stream_wait_limit=%us; aborting MPI job\n",
                     options.rank, options.device, options.output.c_str(), bs, iteration, stage, status,
                     options.kernelTimeout);
        std::fflush(stderr);
        // Do not unwind and free buffers while the failed stream may still use
        // them. Abort the whole MPI job first; the launcher owns process cleanup.
        MPI_Abort(MPI_COMM_WORLD, status);
        std::_Exit(EXIT_FAILURE);
    }
};
using RunVariant = void (*)(const Runtime*);
}  // namespace llbench
#define LL_BENCH_EXPORT extern "C" __attribute__((visibility("default")))
