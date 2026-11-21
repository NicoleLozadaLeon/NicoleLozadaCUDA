#include "Histogram.h"
#include <stdexcept>

namespace hist {
namespace {
inline int widthBin(int value, int width, int bins) { return std::min(value / width, bins - 1); }
}

std::vector<unsigned> byWidthOne(const std::vector<int>& values, int width, int bins) {
    std::vector<unsigned> out(static_cast<std::size_t>(bins), 0);
    for (int v : values) ++out[static_cast<std::size_t>(widthBin(v, width, bins))];
    return out;
}

std::vector<unsigned> byWidthMulti(const std::vector<int>& values, int width, int bins, int threads) {
    return privatizedCpu(values.size(), bins, threads, [&](std::size_t i) { return widthBin(values[i], width, bins); });
}

#ifndef EQ_USE_CUDA
GpuTiming byWidthGpu(const std::vector<int>&, int, int, std::vector<unsigned>&) {
    throw std::runtime_error("built without CUDA");
}
#endif
}
