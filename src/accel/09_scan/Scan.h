#pragma once
#include <vector>
#include "Gpu.h"

namespace scan {
constexpr int kMaxKeys = 256;

struct Grouping {
    std::vector<unsigned> offsets;
    std::vector<int> order;
    bool operator==(const Grouping& o) const { return offsets == o.offsets && order == o.order; }
};

Grouping groupOne(const std::vector<int>& keys, int keyCount);
Grouping groupMulti(const std::vector<int>& keys, int keyCount, int threads);
GpuTiming groupGpu(const std::vector<int>& keys, int keyCount, Grouping& out);
}
