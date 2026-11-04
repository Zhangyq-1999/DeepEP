// Included only by the Python build of ll_cpp_bench/main.cpp. Case, routing,
// Dispatch preparation and the selected Combine binary are shared verbatim.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "runtime/session.hpp"
#include "launch_gate.hpp"

namespace py = pybind11;
namespace llpython {
struct Timer {
    Event first, last;
    void begin(aclrtStream s) { first.record(s); }
    void end(aclrtStream s) { last.record(s); }
    float microseconds() {
        float ms = 0;
        checked(aclrtEventElapsedTime(&ms, first.value, last.value), "read Event");
        if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid Event duration");
        return ms * 1000.0F;
    }
};
class Bench {
    llbench::Options o_;
    std::unique_ptr<deepep::Session> session_;
    llbench::Runtime runtime_{};
    std::unique_ptr<Buffer> comm_;
    std::unique_ptr<Case> case_;
    void live() const { if (!session_ || session_->closed()) throw std::runtime_error("Session is closed"); }
    void hasCase() const { live(); if (!case_) throw std::runtime_error("Prepare a case first"); }
    void sync() const { wait(runtime_.stream, o_.kernelTimeout * 1000U); }
public:
    Bench(int rank, int world, int localRank, const std::vector<std::string>& arguments,
          uintptr_t stream, const std::string& endpoint) {
        auto words = arguments;
        words.insert(words.begin(), "ll-python");
        std::vector<char*> argv;
        for (auto& word : words) argv.push_back(word.data());
        o_ = llbench::Options::parse(int(argv.size()), argv.data(), rank, world, localRank);
        if (!stream || deepep_ll_combine_profile_enabled()) throw std::runtime_error("Use a stream and clean build");
        const auto cores = deepep::query_ll_core_config().active;
        py::gil_scoped_release release;
        session_ = std::make_unique<deepep::Session>(rank, world, llbench::workspaceBytes, endpoint, o_.timeout,
                                                   false, false, false);
        runtime_ = {o_, nullptr, reinterpret_cast<aclrtStream>(stream),
                    static_cast<uint8_t*>(session_->workspace()), static_cast<uint8_t*>(session_->ll_rdma_stage()),
                    session_->capacity(), shmemx_get_ffts_config(), cores, session_->ll_rdma_qps(), nullptr};
        comm_ = std::make_unique<Buffer>(sizeof(DispatchDedup::CommArgs));
        auto args = comm_args(runtime_);
        comm_->upload(&args, sizeof(args));
    }
    // Explicit collective close only; Python error paths terminate the worker
    // without freeing buffers used by peers that may still be running.
    ~Bench() {
        if (session_ && !session_->closed()) {
            case_.release(); comm_.release(); session_.release();
        }
    }
    py::dict identity() const {
        py::dict d;
        d["source_id"] = DEEPEP_BENCH_SOURCE_ID;
        d["original"] = bool(DEEPEP_LL_COMBINE_ORIGINAL);
        d["ranks_per_host"] = o_.ranksPerHost;
        d["active_aiv"] = runtime_.cores;
        d["recv_cores"] = CombineDedup::EpCombineRecvCoreCount(runtime_.cores);
        d["rdma_qps"] = runtime_.rdmaQps;
        d["return_options"] = deepep_ll_combine_return_options();
        d["case_signature"] = o_.signature();
        return d;
    }
    void prepare(uint32_t bs) {
        live();
        if (std::find(o_.tokens.begin(), o_.tokens.end(), bs) == o_.tokens.end())
            throw std::invalid_argument("BS is outside the validated case list");
        py::gil_scoped_release release;
        sync(); case_.reset();
#if DEEPEP_LL_COMBINE_COALESCE > 0
        CombineCoalesce::Layout packed;
        DispatchDedup::EpMemoryServerDedupLayout dl{};
        CombineDedup::Layout cl{};
        const auto recv = CombineDedup::EpCombineRecvCoreCount(runtime_.cores);
        if (!CombineCoalesce::Make(o_.world,o_.ranksPerHost,bs,o_.hidden,o_.topk,o_.experts,runtime_.cores,recv,
                                  DEEPEP_LL_COMBINE_COALESCE==2,packed) ||
            DispatchDedup::BuildDispatchServerDedupLayout(o_.world,bs,o_.hidden,o_.topk,o_.experts,&dl,
                DispatchDedup::EpServerDedupInputMode::Plain16,runtime_.cores)!=0 ||
            !CombineDedup::BuildLayout(o_.world,bs,o_.hidden,o_.topk,o_.experts,cl,runtime_.cores) ||
            dl.data.requiredBytes>CombineCoalesce::kBase || cl.requiredWindowBytes>CombineCoalesce::kBase)
            throw std::runtime_error("Coalescing workspace does not fit");
        for (uint32_t bank=0;bank<2;++bank)
            checked(aclrtMemset(local_base(runtime_)+uint64_t(bank)*DispatchDedup::kEpServerDedupWindowBytes+
                CombineCoalesce::kBase,packed.bytes,0,packed.bytes), "reset coalescing workspace");
#endif
        case_ = std::make_unique<Case>(o_, bs, runtime_.cores, comm_->p);
    }
    void dispatch() {
        hasCase(); py::gil_scoped_release release;
        dispatch_server_dedup_kernel_do_v2(runtime_.stream, &case_->dispatch, case_->variant);
        sync();
    }
    void clearFlags() {
        hasCase(); py::gil_scoped_release release;
        const uint64_t blocks = uint64_t(case_->bs)*o_.topk*((uint64_t(o_.hidden)*2+479)/480);
        combine_clear_result_flags_kernel_do(runtime_.stream, local_base(runtime_), blocks, runtime_.cores);
        sync();
    }
    float sample(py::function callback, bool deviceGate) {
        hasCase();
        Timer timer;
        std::unique_ptr<LaunchGate> gate;
        if (deviceGate) gate = std::make_unique<LaunchGate>(o_.timeout*1000U);
        // All tensor allocation, routing and Host validation precede the gate.
        enqueue(gate.get(), runtime_.stream, [&] { callback(); }, timer, [&] {
            py::gil_scoped_release release;
            combine_server_dedup_kernel_do(runtime_.stream, &case_->combine);
        });
        py::gil_scoped_release release;
        sync();
        return timer.microseconds();
    }
    double precision() { hasCase(); py::gil_scoped_release release; sync(); return case_->precision(o_.dtype=="bf16"); }
    void close() {
        live(); py::gil_scoped_release release; sync(); case_.reset(); comm_.reset(); session_->close(true);
    }
};
}  // namespace llpython

PYBIND11_MODULE(_ll_py_combine, m) {
    m.attr("ranks_per_host") = DispatchDedup::kLlPhysicalRanksPerHost;
    py::class_<llpython::Bench>(m, "Bench")
        .def(py::init<int,int,int,const std::vector<std::string>&,uintptr_t,const std::string&>())
        .def("identity", &llpython::Bench::identity).def("prepare", &llpython::Bench::prepare)
        .def("dispatch", &llpython::Bench::dispatch).def("clear_flags", &llpython::Bench::clearFlags)
        .def("sample", &llpython::Bench::sample).def("precision", &llpython::Bench::precision)
        .def("close", &llpython::Bench::close);
}
