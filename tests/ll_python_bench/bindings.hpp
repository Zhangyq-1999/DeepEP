// Included only by the Python build of ll_cpp_bench/main.cpp. Case, routing,
// Dispatch preparation and the selected Combine binary are shared verbatim.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "runtime/session.hpp"
#include "launch_gate.hpp"

namespace py = pybind11;
extern "C" void ll_bench_entry_sync_do(void* stream, uint64_t ffts);
namespace llpython {
struct SharedSession {
    std::unique_ptr<deepep::Session> session;
    llbench::Runtime runtime{};
    ~SharedSession() {
        // Failed distributed invocations exit without freeing memory peers use.
        if (session && !session->closed()) session.release();
    }
};
using Shared = std::shared_ptr<SharedSession>;
constexpr const char* sessionCapsule = "llpython.shared_session.v1";
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
    Shared shared_;
    bool owner_ = true;
    llbench::Runtime runtime_{};
    std::unique_ptr<Buffer> comm_;
    std::unique_ptr<Case> case_;
#if DEEPEP_BENCH_PROFILE
    std::unique_ptr<Buffer> profileBuffer_;
    std::vector<uint64_t> profileData_;
    llbench::ProfileSamples profiles_;
    std::ofstream profileCsv_;
    uint32_t profileIteration_ = 0;
    bool collecting_ = false;
#endif
    void live() const { if (!shared_ || shared_->session->closed()) throw std::runtime_error("Session is closed"); }
    void hasCase() const { live(); if (!case_) throw std::runtime_error("Prepare a case first"); }
    void sync() const { wait(runtime_.stream, o_.kernelTimeout * 1000U); }
    void initComm() {
#if DEEPEP_LL_OPS_BASE
        if ((o_.hidden != 3584 && o_.hidden != 7168) || o_.topk != 8 || o_.experts != 384)
            throw std::runtime_error("ops-shmem-base supports H3584/H7168, TopK8, E384");
        auto context = context_for(runtime_);
        comm_ = std::make_unique<Buffer>(sizeof(context));
        comm_->upload(&context, sizeof(context));
#else
        comm_ = std::make_unique<Buffer>(sizeof(DispatchDedup::CommArgs));
        auto args = comm_args(runtime_);
        comm_->upload(&args, sizeof(args));
#endif
    }
public:
    Bench(int rank, int world, int localRank, const std::vector<std::string>& arguments,
          uintptr_t stream, const std::string& endpoint) {
        auto words = arguments;
        words.insert(words.begin(), "ll-python");
        std::vector<char*> argv;
        for (auto& word : words) argv.push_back(word.data());
        o_ = llbench::Options::parse(int(argv.size()), argv.data(), rank, world, localRank);
        if (!stream) throw std::runtime_error("Use a valid NPU stream");
        const auto cores = deepep::query_ll_core_config().active;
        py::gil_scoped_release release;
        shared_ = std::make_shared<SharedSession>();
        shared_->session = std::make_unique<deepep::Session>(rank, world, llbench::workspaceBytes, endpoint, o_.timeout,
                                                   false, false, false);
        auto& session_ = shared_->session;
        runtime_ = {o_, nullptr, reinterpret_cast<aclrtStream>(stream),
                    static_cast<uint8_t*>(session_->workspace()), static_cast<uint8_t*>(session_->ll_rdma_stage()),
                    session_->capacity(), shmemx_get_ffts_config(), cores, session_->ll_rdma_qps(), nullptr};
        shared_->runtime = runtime_;
        initComm();
    }
    explicit Bench(py::capsule capsule) : owner_(false) {
        if (!PyCapsule_IsValid(capsule.ptr(), sessionCapsule))
            throw std::invalid_argument("Invalid Python benchmark session capsule");
        shared_ = *capsule.get_pointer<Shared>();
        live();
        runtime_ = shared_->runtime;
        o_ = runtime_.options;
        if (o_.ranksPerHost != DispatchDedup::kLlPhysicalRanksPerHost)
            throw std::runtime_error("Shared session requires matching topology");
        py::gil_scoped_release release;
        initComm();
    }
    py::capsule share() const {
        live();
        return py::capsule(new Shared(shared_), sessionCapsule, [](void* pointer) {
            delete static_cast<Shared*>(pointer);
        });
    }
    void releaseCase() { live(); py::gil_scoped_release release; sync(); case_.reset(); }
    // The caller barriers every rank before and after this untimed phase reset.
    void resetWorkspace() {
        live();
        if (!owner_ || shared_.use_count() != 1 || case_)
            throw std::runtime_error("Release the previous phase before resetting shared workspace");
        py::gil_scoped_release release;
        sync();
        const auto bytes = shared_->session->workspace_allocation_bytes();
        checked(aclrtMemset(runtime_.workspace, bytes, 0, bytes), "reset Python suite workspace");
    }
    // Explicit collective close only; Python error paths terminate the worker
    // without freeing buffers used by peers that may still be running.
    ~Bench() {
        if (shared_ && !shared_->session->closed()) {
            case_.release(); comm_.release();
#if DEEPEP_BENCH_PROFILE
            profileBuffer_.release();
#endif
        }
    }
    py::dict identity() const {
        py::dict d;
        d["source_id"] = DEEPEP_BENCH_SOURCE_ID;
#if DEEPEP_LL_OPS_BASE
        d["implementation"] = "ops-shmem-base";
        d["upstream_revision"] = LL_OPS_REV;
        d["transport"] = "SHMEM per-core QP Put + quiet at upstream batch boundary";
        d["original"] = false;
        d["profile_build"] = false;
        // Upstream has scatter/gather/send roles, followed by all-core final
        // combine. A fixed receive-core count would misdescribe that schedule.
        d["recv_cores"] = 0;
        d["core_layout"] = "upstream-scatter-gather-send-then-all-core-final";
        d["return_options"] = 0;
#else
        d["original"] = bool(DEEPEP_LL_COMBINE_ORIGINAL);
        d["profile_build"] = bool(deepep_ll_combine_profile_enabled());
        d["recv_cores"] = CombineDedup::EpCombineRecvCoreCount(runtime_.cores);
        d["core_layout"] = CombineDedup::kCombineInterleavedCores ? "odd-send-even-receive" : "halves";
        d["control_l2"] = CombineDedup::kCombineControlL2;
        const auto dataL2 = deepep_ll_combine_data_l2();
        if (dataL2 != CombineDedup::kCombineDataL2)
            throw std::runtime_error("Mismatched Host/device data L2 policy; rebuild variant");
        d["data_l2"] = dataL2;
        d["ready_mode"] = DEEPEP_LL_COMBINE_READY;
        d["remote_first"] = bool(deepep_ll_combine_return_options() & 1U);
        d["return_options"] = deepep_ll_combine_return_options();
#endif
        d["ranks_per_host"] = o_.ranksPerHost;
        d["active_aiv"] = runtime_.cores;
        d["rdma_qps"] = runtime_.rdmaQps;
        d["case_signature"] = o_.signature();
        return d;
    }
    void prepare(uint32_t bs) {
        live();
        if (std::find(o_.tokens.begin(), o_.tokens.end(), bs) == o_.tokens.end())
            throw std::invalid_argument("BS is outside the validated case list");
#if DEEPEP_BENCH_PROFILE
        if (collecting_) throw std::runtime_error("Finish the previous profile capture first");
        if (!o_.profile) throw std::runtime_error("Profile build requires --profile 1 or 2");
#endif
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
#if DEEPEP_LL_OPS_BASE
        uint64_t ub = 0;
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        if (ub < 128 * 1024) throw std::runtime_error("Cannot query sufficient UB capacity");
        case_ = std::make_unique<Case>(o_, bs, runtime_.cores, ub, comm_->p);
#else
        case_ = std::make_unique<Case>(o_, bs, runtime_.cores, comm_->p);
#endif
#if DEEPEP_BENCH_PROFILE
        profileData_.assign(size_t(runtime_.cores) * CombineProfile::kWords, 0U);
        profileBuffer_ = std::make_unique<Buffer>(profileData_.size() * sizeof(uint64_t));
        case_->combine.stageTimer = profileBuffer_->p;
        case_->combine.timerMode = o_.profile;
#endif
    }
    void dispatch() {
        hasCase(); py::gil_scoped_release release;
#if !DEEPEP_LL_OPS_BASE
        dispatch_server_dedup_kernel_do_v2(runtime_.stream, &case_->dispatch, case_->variant);
#endif
        sync();
    }
    void clearFlags() {
        hasCase(); py::gil_scoped_release release;
#if !DEEPEP_LL_OPS_BASE
        const uint64_t blocks = uint64_t(case_->bs)*o_.topk*((uint64_t(o_.hidden)*2+479)/480);
        combine_clear_result_flags_kernel_do(runtime_.stream, local_base(runtime_), blocks, runtime_.cores);
#endif
        sync();
    }
    // Enqueue only: the surrounding local Notify is still closed here.
    void entrySync() {
        live(); py::gil_scoped_release release;
        ll_bench_entry_sync_do(runtime_.stream, runtime_.ffts);
    }
    void checkEntrySync() {
        live(); py::gil_scoped_release release;
        ll_bench_entry_sync_do(runtime_.stream, runtime_.ffts);
        sync();
    }
    float sample(py::function callback, bool deviceGate) {
        hasCase();
#if DEEPEP_BENCH_PROFILE
        // Copies and CSV writes stay outside both the gate and the timed Event.
        std::fill(profileData_.begin(), profileData_.end(), 0U);
        profileBuffer_->upload(profileData_.data(), profileBuffer_->bytes);
#endif
        Timer timer;
        std::unique_ptr<LaunchGate> gate;
        if (deviceGate) gate = std::make_unique<LaunchGate>(o_.kernelTimeout*1000U);
        // All tensor allocation, routing and Host validation precede the gate.
        enqueue(gate.get(), runtime_.stream, [&] { callback(); }, timer, [&] {
            py::gil_scoped_release release;
#if DEEPEP_LL_OPS_BASE
            ops_shmem_combine_do(runtime_.stream, &case_->args);
#else
            combine_server_dedup_kernel_do(runtime_.stream, &case_->combine);
#endif
        });
        py::gil_scoped_release release;
        sync();
        const auto elapsed = timer.microseconds();
#if DEEPEP_BENCH_PROFILE
        if (collecting_) {
            profileBuffer_->download(profileData_.data(), profileBuffer_->bytes);
            const auto recv = CombineDedup::EpCombineRecvCoreCount(runtime_.cores);
            llbench::profile_rows(profileCsv_, profileData_, runtime_.cores, recv, o_.rank, o_.world,
                                  case_->bs, o_.device, o_.profile, profileIteration_++);
            profiles_.capture(profileData_, runtime_.cores, recv, o_.profileFrequencyHz != 0);
        }
#endif
        return elapsed;
    }
#if DEEPEP_BENCH_PROFILE
    void beginProfile(const std::string& directory, const std::string& syncMode) {
        hasCase(); sync();
        if (collecting_) throw std::runtime_error("Profile capture already active");
        if (syncMode != "host" && syncMode != "device" && syncMode != "hccl")
            throw std::invalid_argument("Invalid profile synchronization mode");
        o_.output = directory;
        o_.entrySync = syncMode;
        std::filesystem::create_directories(directory);
        const auto path = std::filesystem::path(directory) / ("rank-" + std::to_string(o_.rank) + "-profile.csv");
        const bool exists = std::filesystem::exists(path);
        profileCsv_.open(path, std::ios::app);
        if (!profileCsv_) throw std::runtime_error("Cannot open Python profile output");
        if (!exists) llbench::profile_header(profileCsv_);
        profiles_ = llbench::ProfileSamples{};
        profileIteration_ = 0;
        collecting_ = true;
    }
    void finishProfile(const std::vector<float>& rankMax, py::function gather) {
        hasCase(); sync();
        if (!collecting_ || rankMax.size() != profileIteration_)
            throw std::runtime_error("Profile/Event sample counts differ");
        profileCsv_.close();
        if (!profileCsv_) throw std::runtime_error("Cannot write Python profile output");
        // Gloo only after all measured samples; no MPI initialization in Python.
        struct Peers {
            py::function callback; uint32_t world;
            std::vector<uint8_t> gather(const void* data, size_t bytes) const {
                const auto result = callback(py::bytes(static_cast<const char*>(data), bytes)).cast<std::string>();
                if (result.size() != bytes * world) throw std::runtime_error("Incomplete profile gather");
                return {result.begin(), result.end()};
            }
        } peers{gather, o_.world};
        const auto recv = CombineDedup::EpCombineRecvCoreCount(runtime_.cores);
        case_->exportProducers(o_, peers, runtime_.cores, recv, deepep_ll_combine_return_options());
        profiles_.write(peers, o_, case_->bs, runtime_.cores, recv, rankMax, DEEPEP_BENCH_SOURCE_ID);
        profiles_ = llbench::ProfileSamples{};
        collecting_ = false;
    }
#endif
    void flip_inputs() { hasCase(); py::gil_scoped_release release; sync(); case_->flipExpertSign(o_); }
    double precision() { hasCase(); py::gil_scoped_release release; sync(); return case_->precision(o_.dtype=="bf16"); }
    void close() {
        live();
        if (owner_ && shared_.use_count() != 1)
            throw std::runtime_error("Close borrowed variants before the shared session owner");
        py::gil_scoped_release release;
        sync(); case_.reset(); comm_.reset();
#if DEEPEP_BENCH_PROFILE
        profileBuffer_.reset();
#endif
        if (owner_) shared_->session->close(true);
        shared_.reset();
    }
};
}  // namespace llpython

PYBIND11_MODULE(_ll_py_combine, m) {
    m.attr("suite_api") = 3;
    m.attr("ranks_per_host") = DispatchDedup::kLlPhysicalRanksPerHost;
    py::class_<llpython::Bench>(m, "Bench", py::module_local())
        .def(py::init<int,int,int,const std::vector<std::string>&,uintptr_t,const std::string&>())
        .def(py::init<py::capsule>())
        .def("share", &llpython::Bench::share).def("release_case", &llpython::Bench::releaseCase)
        .def("reset_workspace", &llpython::Bench::resetWorkspace)
        .def("identity", &llpython::Bench::identity).def("prepare", &llpython::Bench::prepare)
        .def("dispatch", &llpython::Bench::dispatch).def("clear_flags", &llpython::Bench::clearFlags)
        .def("entry_sync", &llpython::Bench::entrySync).def("check_entry_sync", &llpython::Bench::checkEntrySync)
        .def("flip_inputs", &llpython::Bench::flip_inputs)
        .def("sample", &llpython::Bench::sample).def("precision", &llpython::Bench::precision)
#if DEEPEP_BENCH_PROFILE
        .def("begin_profile", &llpython::Bench::beginProfile).def("finish_profile", &llpython::Bench::finishProfile)
#endif
        .def("close", &llpython::Bench::close);
}
