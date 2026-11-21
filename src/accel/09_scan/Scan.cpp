#include "Scan.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "Parallel.h"

namespace scan {
Grouping groupOne(const std::vector<int>& keys, int keyCount) {
    Grouping g;
    g.offsets.assign(static_cast<std::size_t>(keyCount) + 1, 0);
    for (int k : keys) ++g.offsets[static_cast<std::size_t>(k) + 1];
    for (std::size_t r = 1; r < g.offsets.size(); ++r) g.offsets[r] += g.offsets[r - 1];
    std::vector<unsigned> next(g.offsets.begin(), g.offsets.end() - 1);
    g.order.resize(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
        g.order[next[static_cast<std::size_t>(keys[i])]++] = static_cast<int>(i);
    return g;
}

Grouping groupMulti(const std::vector<int>& keys, int keyCount, int threads) {
    const std::size_t t = static_cast<std::size_t>(std::max(1, threads));
    const std::size_t r = static_cast<std::size_t>(keyCount);
    std::vector<unsigned> counts(r * t, 0);
    parallelFor(keys.size(), static_cast<int>(t), [&](std::size_t begin, std::size_t end, std::size_t id) {
        std::vector<unsigned> local(r, 0);
        for (std::size_t i = begin; i < end; ++i) ++local[static_cast<std::size_t>(keys[i])];
        for (std::size_t k = 0; k < r; ++k) counts[k * t + id] = local[k];
    });
    unsigned running = 0;
    for (unsigned& c : counts) {
        const unsigned here = c;
        c = running;
        running += here;
    }
    Grouping g;
    g.offsets.resize(r + 1);
    for (std::size_t k = 0; k < r; ++k) g.offsets[k] = counts[k * t];
    g.offsets[r] = static_cast<unsigned>(keys.size());
    g.order.resize(keys.size());
    parallelFor(keys.size(), static_cast<int>(t), [&](std::size_t begin, std::size_t end, std::size_t id) {
        std::vector<unsigned> next(r);
        for (std::size_t k = 0; k < r; ++k) next[k] = counts[k * t + id];
        for (std::size_t i = begin; i < end; ++i) g.order[next[static_cast<std::size_t>(keys[i])]++] = static_cast<int>(i);
    });
    return g;
}

#ifndef EQ_USE_CUDA
GpuTiming groupGpu(const std::vector<int>&, int, Grouping&) { throw std::runtime_error("built without CUDA"); }
#endif
}
