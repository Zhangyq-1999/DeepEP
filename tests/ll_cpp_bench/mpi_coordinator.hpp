#pragma once
#include <mpi.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../csrc/kernels/dispatch/low_latency/common/ll_group_policy.h"

namespace llbench {
inline void mpi_check(int status, const char* operation)
{
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
}
struct Coordinator {
    int rank = 0, world = 0, localRank = 0;
    Coordinator()
    {
        mpi_check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI error handler");
        mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI rank");
        mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &world), "MPI size");
        if (world != 2 && world != 4 && world != 8 && world != 16 && world != 32 && world != 64)
            throw std::runtime_error("Use 2, 4, 8, 16, 32 or 64 MPI ranks");
        MPI_Comm local = MPI_COMM_NULL;
        mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local),
                  "MPI local group");
        int size = 0, first = 0;
        mpi_check(MPI_Comm_rank(local, &localRank), "MPI local rank");
        mpi_check(MPI_Comm_size(local, &size), "MPI local size");
        first = rank;
        mpi_check(MPI_Bcast(&first, 1, MPI_INT, 0, local), "MPI first local rank");
        mpi_check(MPI_Comm_free(&local), "MPI free local group");
        const int hostRanks = int(DispatchDedup::kLlPhysicalRanksPerHost);
        const int perHost = world < hostRanks ? world : hostRanks;
        if (size != perHost || first % perHost || rank != first + localRank)
            throw std::runtime_error("MPI host layout does not match the compiled ranks-per-host; "
                                     "rebuild all variants for this topology");
    }
    void barrier() const
    {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI barrier");
    }
    std::vector<uint8_t> gather(const void* value, size_t bytes) const
    {
        if (bytes > size_t(std::numeric_limits<int>::max())) throw std::overflow_error("MPI message too large");
        std::vector<uint8_t> result(bytes * size_t(world));
        mpi_check(MPI_Allgather(value, int(bytes), MPI_BYTE, result.data(), int(bytes), MPI_BYTE, MPI_COMM_WORLD),
                  "MPI allgather");
        return result;
    }
    void verify(uint64_t identity) const
    {
        std::vector<uint64_t> values(world);
        mpi_check(MPI_Allgather(&identity, 1, MPI_UINT64_T, values.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD),
                  "MPI configuration");
        for (auto value : values)
            if (value != identity) throw std::runtime_error("Ranks use different builds or benchmark configurations");
    }
};
}  // namespace llbench
