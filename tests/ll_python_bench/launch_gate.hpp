#pragma once
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace llpython {
inline void checked(int status, const char* what) {
    if (status) throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
}
inline void wait(aclrtStream stream, unsigned timeoutMs) {
    const int status = aclrtSynchronizeStreamWithTimeout(stream, int32_t(timeoutMs));
    if (status) {
        // Never free live inputs or run collective destructors after a device error.
        std::fprintf(stderr, "FAIL Python Combine stream status=%d timeout_ms=%u\n", status, timeoutMs);
        std::fflush(stderr);
        std::_Exit(1);
    }
}
// Adapted from Ziha0-Zeng/DeepEP b1ed463: a local Notify holds the stream
// until the SHMEM barrier (or optional HCCL dependency), Events and native
// launch have ALL been submitted. The local gate itself uses ACL Notify.
class LaunchGate {
    aclrtNotify notify_ = nullptr;
    aclrtStream control_ = nullptr, target_ = nullptr;
    unsigned timeoutMs_;
    bool armed_ = false, released_ = false;
public:
    explicit LaunchGate(unsigned timeoutMs) : timeoutMs_(timeoutMs) {
        checked(aclrtCreateNotify(&notify_, 0), "create Notify");
        try { checked(aclrtCreateStream(&control_), "create release stream"); }
        catch (...) { aclrtDestroyNotify(notify_); throw; }
    }
    LaunchGate(const LaunchGate&) = delete;
    ~LaunchGate() {
        if (armed_ && !released_) {
            // Callback/launch error: stop this worker; torchrun owns job teardown.
            // Releasing a partially submitted collective here can deadlock peers.
            std::fputs("FAIL benchmark gate did not release; exiting worker\n", stderr);
            std::_Exit(1);
        }
        if (armed_) wait(target_, timeoutMs_);
        if (control_) { wait(control_, timeoutMs_); aclrtDestroyStream(control_); }
        if (notify_) aclrtDestroyNotify(notify_);
    }
    void arm(aclrtStream stream) {
        target_ = stream;
        checked(aclrtWaitAndResetNotify(notify_, stream, timeoutMs_), "arm Notify");
        armed_ = true;
    }
    void release() {
        checked(aclrtRecordNotify(notify_, control_), "release Notify");
        released_ = true;
    }
};

template<class Gate, class Callback, class Timer, class Launch>
void enqueue(Gate* gate, aclrtStream stream, Callback&& callback, Timer& timer, Launch&& launch) {
    if (gate) gate->arm(stream);
    callback();                    // SHMEM launch, or HCCL Work.wait + queue drain
    timer.begin(stream);
    launch();
    timer.end(stream);
    if (gate) gate->release();     // Nothing timed remains to be submitted
}
}  // namespace llpython
