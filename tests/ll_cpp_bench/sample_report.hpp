#pragma once
#include "options.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

namespace llbench {
// All ranks have the same gathered Event vector. Preserve sample order and
// the winning rank; outlier analysis must operate on whole collective rounds.
inline std::vector<float> rank_max_samples(const std::vector<uint8_t>& data, const Options& o, uint32_t bs)
{
    if (data.size() != size_t(o.world) * o.iterations * sizeof(float))
        throw std::runtime_error("Incomplete global Event samples");
    std::vector<float> result(o.iterations);
    std::vector<uint32_t> owners(o.iterations);
    for (uint32_t r = 0; r < o.world; ++r)
        for (uint32_t i = 0; i < o.iterations; ++i) {
            float value;
            std::memcpy(&value, data.data() + (size_t(r) * o.iterations + i) * sizeof(float), sizeof(float));
            if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Invalid global Event sample");
            if (value > result[i]) { result[i] = value; owners[i] = r; }
        }
    if (!o.rank) {
        const auto path = std::filesystem::path(o.output) / "rank-max.csv";
        const bool exists = std::filesystem::exists(path);
        std::ofstream out(path, std::ios::app);
        if (!exists) out << "tokens,iteration,rank,us\n";
        // Preserve the float ordering used to select median/worst diagnostics.
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (uint32_t i = 0; i < o.iterations; ++i) out << bs << ',' << i << ',' << owners[i] << ',' << result[i] << '\n';
        out.close();
        if (!out) throw std::runtime_error("Cannot write global Event samples");
    }
    return result;
}
}  // namespace llbench
