#pragma once
#include "variant_api.hpp"
#include <stdexcept>

extern "C" void ll_bench_entry_sync_do(void* stream, uint64_t ffts, uint64_t waitTicks);

namespace llbench {
// Queue synchronization, Events and Combine before releasing the local stream.
// Otherwise a fast device could finish its barrier before Host submits Combine.
class EntrySync {
    const Runtime& runtime_;
    aclrtNotify notify_ = nullptr;
    aclrtStream control_ = nullptr;
    bool armed_ = false;
    static void check(int status, const char* operation)
    {
        if (status) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
public:
    explicit EntrySync(const Runtime& runtime) : runtime_(runtime)
    {
        if (runtime_.options.entrySync == "host") return;
        check(aclrtCreateNotify(&notify_, 0), "create entry Notify");
        const int status = aclrtCreateStream(&control_);
        if (status) {
            aclrtDestroyNotify(notify_);
            notify_ = nullptr;
            check(status, "create entry control stream");
        }
    }
    EntrySync(const EntrySync&) = delete;
    ~EntrySync()
    {
        // On submission failure unblock the stream before the outer MPI abort;
        // never release resources while an armed wait can still reference them.
        if (armed_) {
            if (aclrtRecordNotify(notify_, control_)) {
                MPI_Abort(MPI_COMM_WORLD, 1);
                std::_Exit(EXIT_FAILURE);
            }
        }
        if (notify_) runtime_.synchronize("entry sync cleanup");
        if (control_) {
            const int status = aclrtSynchronizeStreamWithTimeout(
                control_, int32_t(runtime_.options.kernelTimeout * 1000U));
            if (status) { MPI_Abort(MPI_COMM_WORLD, status); std::_Exit(EXIT_FAILURE); }
            aclrtDestroyStream(control_);
        }
        if (notify_) aclrtDestroyNotify(notify_);
    }
    void before()
    {
        if (!notify_) { runtime_.peers->barrier(); return; }
        check(aclrtWaitAndResetNotify(notify_, runtime_.stream,
              runtime_.options.kernelTimeout * 1000U), "arm entry Notify");
        armed_ = true;
        // Ascend950 system counter: 1 GHz. Host stream timeout remains the
        // wall-clock bound, including SDK CQ polling and missing peers.
        ll_bench_entry_sync_do(runtime_.stream, runtime_.ffts,
                               uint64_t(runtime_.options.kernelTimeout) * 1000000000ULL);
    }
    void submitted()
    {
        if (!notify_) return;
        check(aclrtRecordNotify(notify_, control_), "release entry Notify");
        armed_ = false;
    }
};
}  // namespace llbench
