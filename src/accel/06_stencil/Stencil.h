#pragma once
#include "Gpu.h"
#include "MapGrid.h"

namespace stencil {
constexpr float kDecay = 0.98f;
constexpr double kTolerance = 1e-6;

Grid cpuOne(const Grid& sources, int sweeps);
Grid cpuMulti(const Grid& sources, int sweeps, int threads);
GpuTiming gpu(const Grid& sources, int sweeps, Grid& out);
}
