#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
#include "Gpu.h"
#include "Parallel.h"

namespace hist {
constexpr int kMaxSharedBins = 12288;

template <class BinOf>
std::vector<unsigned> privatizedCpu(std::size_t n, int bins, int threads, BinOf binOf) {
    const std::size_t t = static_cast<std::size_t>(std::max(1, threads));
    std::vector<std::vector<unsigned>> local(t, std::vector<unsigned>(static_cast<std::size_t>(bins), 0));
    parallelFor(n, static_cast<int>(t), [&](std::size_t begin, std::size_t end, std::size_t id) {
        std::vector<unsigned>& mine = local[id];
        for (std::size_t i = begin; i < end; ++i) ++mine[static_cast<std::size_t>(binOf(i))];
    });
    std::vector<unsigned> out(static_cast<std::size_t>(bins), 0);
    for (const std::vector<unsigned>& part : local)
        for (std::size_t b = 0; b < out.size(); ++b) out[b] += part[b];
    return out;
}

std::vector<unsigned> byWidthOne(const std::vector<int>& values, int width, int bins);
std::vector<unsigned> byWidthMulti(const std::vector<int>& values, int width, int bins, int threads);
GpuTiming byWidthGpu(const std::vector<int>& values, int width, int bins, std::vector<unsigned>& out);
}
