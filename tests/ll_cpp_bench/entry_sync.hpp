#pragma once
#include "variant_api.hpp"

extern "C" void ll_bench_entry_sync_do(void* stream, uint64_t ffts);

namespace llbench {
// A local Notify holds the stream until the barrier AND measured work are
// queued. Without it, a fast barrier can expose Host launch gaps to the Event.
class EntrySync {
    const Runtime& runtime_;
    aclrtNotify notify_ = nullptr;
    aclrtStream release_ = nullptr;
    bool pending_ = false;
    [[noreturn]] void fail(const char* what, int status) const
    {
        std::fprintf(stderr, "FAIL rank=%u device=%u entry_sync=%s stage=%s status=%d\n",
                     runtime_.options.rank, runtime_.options.device,
                     runtime_.options.entrySync.c_str(), what, status);
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, status ? status : 1);
        std::_Exit(EXIT_FAILURE);
    }
    void check(int status, const char* what) const { if (status) fail(what, status); }
public:
    explicit EntrySync(const Runtime& runtime) : runtime_(runtime)
    {
        if (runtime_.options.entrySync == "host") return;
        check(aclrtCreateNotify(&notify_, 0), "create Notify");
        check(aclrtCreateStream(&release_), "create release stream");
    }
    EntrySync(const EntrySync&) = delete;
    ~EntrySync()
    {
        if (pending_) runtime_.synchronize("entry sync cleanup");
        if (release_) {
            check(aclrtSynchronizeStreamWithTimeout(release_, int32_t(runtime_.options.kernelTimeout * 1000U)),
                  "release stream completion");
            aclrtDestroyStream(release_);
        }
        if (notify_) aclrtDestroyNotify(notify_);
    }
    template<class Work> void enqueue(Work&& work)
    {
        if (!notify_) {
            runtime_.peers->barrier();
            work();
            return;
        }
        if (pending_) fail("previous invocation not completed", 1);
        try {
            check(aclrtWaitAndResetNotify(notify_, runtime_.stream, runtime_.options.kernelTimeout * 1000U),
                  "arm Notify");
            pending_ = true;
            ll_bench_entry_sync_do(runtime_.stream, runtime_.ffts);
            work();
            check(aclrtRecordNotify(notify_, release_), "release Notify");
        } catch (...) {
            // Never unwind and free inputs with a half-submitted collective.
            fail("submit barrier and work", 1);
        }
    }
    void complete(const char* stage, unsigned bs = 0, unsigned iteration = 0)
    {
        runtime_.synchronize(stage, bs, iteration);
        pending_ = false;
    }
};
} // namespace llbench
