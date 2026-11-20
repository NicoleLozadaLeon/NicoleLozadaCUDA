#pragma once
#include <vector>
#include "Gpu.h"
#include "MapGrid.h"

namespace conv {
constexpr int kMaxRadius = 6;
constexpr double kTolerance = 1e-5;

std::vector<float> gaussian(int radius);
Grid cpuOne(const Grid& in, const std::vector<float>& filter, int radius);
Grid cpuMulti(const Grid& in, const std::vector<float>& filter, int radius, int threads);
GpuTiming gpu(const Grid& in, const std::vector<float>& filter, int radius, Grid& out);
}
