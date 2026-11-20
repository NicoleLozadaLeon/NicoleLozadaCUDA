#include "Nearby.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "Parallel.h"

namespace nearby {
namespace {
bool close(int ax, int ay, int bx, int by, int r2) {
    const int dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy <= r2;
}
}

// Reference: every corner against every courier.
std::vector<int> cpuOne(const Points& corners, const Points& couriers, int radiusM) {
    const int r2 = radiusM * radiusM;
    std::vector<int> out(corners.x.size(), 0);
    for (std::size_t i = 0; i < corners.x.size(); ++i) {
        int count = 0;
        for (std::size_t j = 0; j < couriers.x.size(); ++j)
            if (close(corners.x[i], corners.y[i], couriers.x[j], couriers.y[j], r2)) ++count;
        out[i] = count;
    }
    return out;
}

// Best CPU version: couriers bucketed into square cells of side radiusM, so a corner only looks at
// the 3 x 3 cells around its own (a courier within the radius can be at most one cell away).
std::vector<int> cpuMulti(const Points& corners, const Points& couriers, int radiusM, int threads) {
    const int r2 = radiusM * radiusM;
    int maxX = 0, maxY = 0;
    for (const Points* p : {&corners, &couriers})
        for (std::size_t i = 0; i < p->x.size(); ++i) {
            maxX = std::max(maxX, p->x[i]);
            maxY = std::max(maxY, p->y[i]);
        }
    const int gw = maxX / radiusM + 1, gh = maxY / radiusM + 1;
    auto cellOf = [&](int x, int y) { return static_cast<std::size_t>((y / radiusM) * gw + x / radiusM); };

    std::vector<int> start(static_cast<std::size_t>(gw) * gh + 1, 0);
    for (std::size_t j = 0; j < couriers.x.size(); ++j) ++start[cellOf(couriers.x[j], couriers.y[j]) + 1];
    for (std::size_t c = 1; c < start.size(); ++c) start[c] += start[c - 1];
    std::vector<int> next(start.begin(), start.end() - 1), bx(couriers.x.size()), by(couriers.x.size());
    for (std::size_t j = 0; j < couriers.x.size(); ++j) {
        const std::size_t slot = static_cast<std::size_t>(next[cellOf(couriers.x[j], couriers.y[j])]++);
        bx[slot] = couriers.x[j];
        by[slot] = couriers.y[j];
    }

    std::vector<int> out(corners.x.size(), 0);
    parallelFor(corners.x.size(), threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t i = begin; i < end; ++i) {
            const int x = corners.x[i], y = corners.y[i], cx = x / radiusM, cy = y / radiusM;
            int count = 0;
            for (int gy = std::max(0, cy - 1); gy <= std::min(gh - 1, cy + 1); ++gy)
                for (int gx = std::max(0, cx - 1); gx <= std::min(gw - 1, cx + 1); ++gx) {
                    const std::size_t c = static_cast<std::size_t>(gy * gw + gx);
                    for (int k = start[c]; k < start[c + 1]; ++k)
                        if (close(x, y, bx[static_cast<std::size_t>(k)], by[static_cast<std::size_t>(k)], r2)) ++count;
                }
            out[i] = count;
        }
    });
    return out;
}

#ifndef EQ_USE_CUDA
DeviceCorners::DeviceCorners(const Points&) { throw std::runtime_error("built without CUDA"); }
DeviceCorners::~DeviceCorners() = default;
GpuTiming gpuTiled(const DeviceCorners&, const Points&, int, std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
GpuTiming gpuNaive(const DeviceCorners&, const Points&, int, std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
GpuTiming gpuUploadAll(const Points&, const Points&, int, std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
#endif
}
