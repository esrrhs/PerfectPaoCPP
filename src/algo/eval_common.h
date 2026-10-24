// Shared helpers for the evaluation harnesses (CSV append, snapshot meta,
// rolling snapshot pruning). Used by eval_rule and eval_pimc.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

namespace algo {
namespace eval_common {

inline std::string timestampNow() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::tm tmNow{};
#if defined(_WIN32)
    localtime_s(&tmNow, &t);
#else
    localtime_r(&t, &tmNow);
#endif
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmNow);
    return buf;
}

// Appends one CSV row, writing the header first for a fresh file.
inline void appendCsv(const std::string& path, const std::string& header,
                      const std::string& row) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    bool needHeader = true;
    if (FILE* probe = std::fopen(path.c_str(), "rb")) {
        std::fseek(probe, 0, SEEK_END);
        needHeader = (std::ftell(probe) <= 0);
        std::fclose(probe);
    }
    if (FILE* f = std::fopen(path.c_str(), "ab")) {
        if (needHeader) std::fprintf(f, "%s\n", header.c_str());
        std::fprintf(f, "%s\n", row.c_str());
        std::fclose(f);
    }
}

// Keeps only the `maxSnapshots` most recent u<update> directories under root.
inline void pruneSnapshots(const std::string& root, int maxSnapshots) {
    if (maxSnapshots <= 0 || root.empty()) return;
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return;
    std::vector<std::pair<long long, std::filesystem::path>> dirs;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < 2 || name[0] != 'u') continue;
        long long upd = 0;
        try {
            upd = std::stoll(name.substr(1));
        } catch (...) {
            continue;
        }
        dirs.emplace_back(upd, entry.path());
    }
    if (static_cast<int>(dirs.size()) > maxSnapshots) {
        std::sort(dirs.begin(), dirs.end());
        int toRemove = static_cast<int>(dirs.size()) - maxSnapshots;
        for (int i = 0; i < toRemove; ++i) {
            std::filesystem::remove_all(dirs[static_cast<size_t>(i)].second, ec);
        }
    }
}

}  // namespace eval_common
}  // namespace algo
