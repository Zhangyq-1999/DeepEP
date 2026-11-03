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
    void synchronize(const char* stage, unsigned bs = 0, unsigned iteration = 0,
                     aclrtEvent start = nullptr, aclrtEvent end = nullptr) const
    {
        const int status = aclrtSynchronizeStreamWithTimeout(stream, int32_t(options.kernelTimeout * 1000U));
        if (!status) return;
        std::fprintf(stderr, "FAIL rank=%u device=%u output=%s BS=%u iteration=%u stage=%s status=%d "
                             "stream_wait_limit=%us; aborting MPI job\n",
                     options.rank, options.device, options.output.c_str(), bs, iteration, stage, status,
                     options.kernelTimeout);
        std::fflush(stderr);
        // Only sampling supplies Events recorded in this iteration. Query them
        // after a failed wait; add no tasks or waits to the measured path.
        if (start && end) {
            aclrtEventRecordedStatus first = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            aclrtEventRecordedStatus last = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            const int firstRc = aclrtQueryEventStatus(start, &first);
            const int lastRc = aclrtQueryEventStatus(end, &last);
            const char* boundary = "unknown";
            if (!firstRc && !lastRc) {
                if (last == ACL_EVENT_RECORDED_STATUS_COMPLETE)
                    boundary = "end_event_complete";
                else if (last == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
                    if (first == ACL_EVENT_RECORDED_STATUS_COMPLETE) boundary = "between_start_and_end";
                    else if (first == ACL_EVENT_RECORDED_STATUS_NOT_READY) boundary = "before_start_event";
                }
            }
            // A completed start proves local entry sync passed. An incomplete
            // start does not distinguish Notify, entry kernel and Event itself.
            // A runtime query failure must never be interpreted as completion.
            std::fprintf(stderr, "FAIL_BOUNDARY rank=%u BS=%u iteration=%u entry_sync=%s boundary=%s "
                                 "start_rc=%d start_status=%d end_rc=%d end_status=%d "
                                 "(status:0=not_ready,1=complete; valid only when rc=0)\n",
                         options.rank, bs, iteration, options.entrySync.c_str(), boundary,
                         firstRc, int(first), lastRc, int(last));
            std::fflush(stderr);
        }
        // Do not unwind and free buffers while the failed stream may still use
        // them. Abort the whole MPI job first; the launcher owns process cleanup.
        MPI_Abort(MPI_COMM_WORLD, status);
        std::_Exit(EXIT_FAILURE);
    }
};
using RunVariant = void (*)(const Runtime*);
}  // namespace llbench
#define LL_BENCH_EXPORT extern "C" __attribute__((visibility("default")))
