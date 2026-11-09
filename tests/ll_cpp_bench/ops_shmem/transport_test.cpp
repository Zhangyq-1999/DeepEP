#include <cstdint>
#define GM_ADDR uint8_t*
#include "transport.hpp"
#include <cassert>
#include <iostream>

int main()
{
    uint8_t local[8192]{}, remote[8192]{}, scratch[512]{};
    Model::local = local;
    Model::remote = remote;
    AscendC::core = 53;
    std::fill(local + 512, local + 1024, 0x21);
    std::fill(local + 1024, local + 1536, 0x43);
    opsbench::Transport transport;
#ifdef DEEPEP_LL_COMBINE_PROFILE
    std::vector<uint64_t> records(56 * CombineProfile::kWords);
    CombineProfile::DeviceProfile profile;
    profile.Init(reinterpret_cast<uint8_t*>(records.data()), 2);
    transport.SetProfile(&profile);
#endif
    transport.Init({scratch}, sizeof(scratch));
    auto batch = transport.MakeBatchHandle(17, {scratch}, sizeof(scratch), local, local);
    transport.WriteNbi(batch, local + 4096, local + 512, 512);
    transport.WriteNbi(batch, local + 4608, local + 1024, 512);
    assert(Model::pending.size() == 2 && Model::quiets == 0);
    for (const auto& request : Model::pending) {
        assert(request.peer == 17 && request.qp == 53);
        assert(request.src >= local + 512 && request.src < local + 1536);
        assert(request.dst >= remote + 4096 && request.dst < remote + 5120);
    }
    transport.BatchCommit(batch);
    assert(Model::pending.empty() && Model::quiets == 1);
    assert(std::memcmp(remote + 4096, local + 512, 1024) == 0);
    transport.BatchCommit(batch);
    assert(Model::quiets == 1);  // empty batch has no completion to retire
#ifdef DEEPEP_LL_COMBINE_PROFILE
    profile.Mark(CombineProfile::FlushDone);
    profile.Finish(1, 32, 2);
    using namespace CombineProfile;
    const auto* row = records.data() + 53 * kWords;
    assert(row[RdmaPackets] == 2 && row[RdmaBytes] == 1024);
    assert(row[RdmaQuietCalls] == 1);
    assert((row[Valid] & (1ULL << RdmaFirstPut)) && (row[Valid] & (1ULL << RdmaLastPut)));
#endif
    std::cout << "SHMEM adapter: symmetric translation, per-core QP and batch completion passed\n";
}
