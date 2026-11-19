#pragma once
#include <vector>
#include "Eta.h"
#include "Gpu.h"

namespace argmin {
using Key = unsigned long long;
constexpr Key kNone = ~0ULL;

inline Key pack(int eta, int id) {
    return (static_cast<Key>(static_cast<unsigned>(eta)) << 32) | static_cast<unsigned>(id);
}
inline int etaOf(Key key) { return static_cast<int>(key >> 32); }
inline int idOf(Key key) { return static_cast<int>(key & 0xffffffffULL); }

std::vector<Key> cpuOne(const std::vector<int>& eta, int rows, int cols, const std::vector<int>& ids);
std::vector<Key> cpuMulti(const std::vector<int>& eta, int rows, int cols, const std::vector<int>& ids, int threads);
GpuTiming gpu(const std::vector<int>& eta, int rows, int cols, const std::vector<int>& ids, std::vector<Key>& out);

Key bestCourierGpu(const eta::DeviceTable& table, const eta::Params& p, const eta::Query& q,
                   const eta::Couriers& couriers, const std::vector<int>& ids);
}
