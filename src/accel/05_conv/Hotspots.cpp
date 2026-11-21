#include "Hotspots.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "Parallel.h"

namespace hotspots {
namespace {
constexpr float kSuppressed = -1.0f;

void smoothRows(const Grid& in, const std::vector<float>& filter, int y0, int y1, Grid& out) {
    for (int y = y0; y < y1; ++y)
        for (int x = 0; x < in.w; ++x) {
            float sum = 0.0f;
            for (int fy = 0; fy < kSide; ++fy) {
                const int yy = y - kRadius + fy;
                if (yy < 0 || yy >= in.h) continue;
                for (int fx = 0; fx < kSide; ++fx) {
                    const int xx = x - kRadius + fx;
                    if (xx < 0 || xx >= in.w) continue;
                    sum += filter[static_cast<std::size_t>(fy * kSide + fx)] * in.at(xx, yy);
                }
            }
            out.at(x, y) = sum;
        }
}

std::size_t indexOfMaximum(const Grid& g) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < g.v.size(); ++i)
        if (g.v[i] > g.v[best]) best = i;  // strictly greater: the lowest cell wins a tie
    return best;
}

void suppressAround(Grid& g, int cx, int cy, int radius) {
    for (int y = std::max(0, cy - radius); y <= std::min(g.h - 1, cy + radius); ++y)
        for (int x = std::max(0, cx - radius); x <= std::min(g.w - 1, cx + radius); ++x) g.at(x, y) = kSuppressed;
}
}

std::vector<float> gaussian() {
    std::vector<float> f(static_cast<std::size_t>(kSide * kSide));
    const double twoSigmaSquared = 2.0 * kSigmaCells * kSigmaCells;
    double total = 0.0;
    for (int y = -kRadius; y <= kRadius; ++y)
        for (int x = -kRadius; x <= kRadius; ++x) total += std::exp(-(x * x + y * y) / twoSigmaSquared);
    for (int y = -kRadius; y <= kRadius; ++y)
        for (int x = -kRadius; x <= kRadius; ++x)
            f[static_cast<std::size_t>((y + kRadius) * kSide + x + kRadius)] =
                static_cast<float>(std::exp(-(x * x + y * y) / twoSigmaSquared) / total);
    return f;
}

Grid countPoints(int w, int h, const std::vector<int>& cells) {
    Grid g(w, h);
    for (int c : cells)
        if (c >= 0 && static_cast<std::size_t>(c) < g.v.size()) g.v[static_cast<std::size_t>(c)] += 1.0f;
    return g;
}

Grid cpuOne(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter) {
    const Grid counts = countPoints(w, h, cells);
    Grid out(w, h);
    smoothRows(counts, filter, 0, h, out);
    return out;
}

// The counting stays sequential (a few thousand orders); the smoothing splits the rows between the threads.
Grid cpuMulti(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, int threads) {
    const Grid counts = countPoints(w, h, cells);
    Grid out(w, h);
    parallelFor(static_cast<std::size_t>(h), threads, [&](std::size_t y0, std::size_t y1, std::size_t) {
        smoothRows(counts, filter, static_cast<int>(y0), static_cast<int>(y1), out);
    });
    return out;
}

std::vector<Hotspot> rank(const Grid& smoothed, double cellM, const std::vector<Cell>& couriers, int count,
                          int minSeparationCells) {
    std::vector<Hotspot> zones;
    if (smoothed.v.empty()) return zones;
    Grid candidates = smoothed;
    for (int n = 0; n < count; ++n) {
        const std::size_t best = indexOfMaximum(candidates);
        if (candidates.v[best] <= 0.0f) break;  // no demand left
        Hotspot z;
        z.x = static_cast<int>(best % static_cast<std::size_t>(smoothed.w));
        z.y = static_cast<int>(best / static_cast<std::size_t>(smoothed.w));
        z.value = smoothed.v[best];
        double bestSq = 0.0;
        for (std::size_t c = 0; c < couriers.size(); ++c) {
            const double dx = couriers[c].x - z.x, dy = couriers[c].y - z.y;
            const double sq = dx * dx + dy * dy;
            if (z.courier < 0 || sq < bestSq) {  // strictly less: the lowest courier wins a tie
                z.courier = static_cast<int>(c);
                bestSq = sq;
            }
        }
        if (z.courier >= 0) z.distanceM = std::sqrt(bestSq) * cellM;
        z.uncovered = z.courier < 0 || z.distanceM > kFarMetres;
        zones.push_back(z);
        suppressAround(candidates, z.x, z.y, minSeparationCells);
    }
    return zones;
}

#ifndef EQ_USE_CUDA
GpuTiming gpuTiled(int, int, const std::vector<int>&, const std::vector<float>&, Grid&) {
    throw std::runtime_error("built without CUDA");
}
GpuTiming gpuDirect(int, int, const std::vector<int>&, const std::vector<float>&, Grid&) {
    throw std::runtime_error("built without CUDA");
}
void gpuWarmUp(int) { throw std::runtime_error("built without CUDA"); }
#endif
}
