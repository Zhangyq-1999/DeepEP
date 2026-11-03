#pragma once
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace llbench {
// One status file per host. Updated outside Event timing and before the next
// MPI barrier; the launcher's monitor runs in a separate Host process.
struct Progress {
    std::filesystem::path path;
    std::string variant;
    int rank;
    Progress(const std::string& root, int r, int localRank, int perHost) : rank(r)
    {
        if (localRank != 0) return;
        path = std::filesystem::path(root) / ("node-" + std::to_string(r / perHost)) / "progress.txt";
        std::filesystem::create_directories(path.parent_path());
    }
    void report(const char* stage, unsigned bs = 0, unsigned done = 0, unsigned total = 0) const
    {
        if (path.empty()) return;
        auto temporary = path;
        temporary += ".tmp";
        std::ofstream file(temporary);
        file << "rank=" << rank;
        if (!variant.empty()) file << " variant=" << variant;
        if (bs) file << " BS=" << bs;
        file << " stage=" << stage;
        if (total) file << " completed=" << done << '/' << total;
        file << '\n';
        file.close();
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) std::fprintf(stderr, "Progress file update failed: %s\n", error.message().c_str());
    }
};
} // namespace llbench
