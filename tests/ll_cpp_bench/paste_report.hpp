#pragma once
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Keep every record. Prefer whole BS sections; an unusually large single case
// is continued at a line boundary. Each part repeats the decoding legend.
inline void WritePasteReport(std::ostream& out, const std::string& report,
                             size_t limit = 62U * 1024U)
{
    if (report.size() <= limit) { out << report; return; }
    const auto first = report.find("\nBS=");
    if (first == std::string::npos || first + 256U >= limit)
        throw std::runtime_error("Compact report has no bounded case header");
    const auto header = report.substr(0, first);
    const auto capacity = limit - header.size() - 256U;
    std::vector<std::string> parts;
    std::string current;
    for (size_t begin = first; begin < report.size();) {
        auto end = report.find("\nBS=", begin + 1U);
        if (end == std::string::npos) end = report.size();
        // Start a case on a fresh part if it cannot fit alongside prior cases.
        if (!current.empty() && end - begin > capacity - current.size()) {
            parts.push_back(std::move(current)); current.clear();
        }
        while (end - begin > capacity) {
            auto cut = report.rfind('\n', begin + capacity - 1U);
            cut = cut == std::string::npos || cut < begin ? begin + capacity : cut + 1U;
            parts.push_back(report.substr(begin, cut - begin));
            begin = cut;
        }
        current.append(report, begin, end - begin);
        begin = end;
    }
    if (!current.empty()) parts.push_back(std::move(current));
    for (size_t i = 0; i < parts.size(); ++i) {
        out << "LL_COMBINE PASTE_BEGIN part=" << i + 1U << '/' << parts.size() << '\n'
            << header << parts[i]
            << "LL_COMBINE PASTE_END part=" << i + 1U << '/' << parts.size() << '\n';
    }
}
