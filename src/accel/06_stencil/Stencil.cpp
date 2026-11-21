#include "Stencil.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include "Parallel.h"

namespace stencil {
namespace {
Grid initial(const Grid& sources) {
    Grid u(sources.w, sources.h);
    for (std::size_t i = 0; i < u.v.size(); ++i) u.v[i] = sources.v[i] > 0.0f ? 1.0f : 0.0f;
    return u;
}

void sweepRows(const Grid& sources, const Grid& u, Grid& next, int y0, int y1) {
    for (int y = y0; y < y1; ++y)
        for (int x = 0; x < u.w; ++x) {
            if (sources.at(x, y) > 0.0f) {
                next.at(x, y) = 1.0f;
                continue;
            }
            const float up = y > 0 ? u.at(x, y - 1) : 0.0f;
            const float down = y + 1 < u.h ? u.at(x, y + 1) : 0.0f;
            const float left = x > 0 ? u.at(x - 1, y) : 0.0f;
            const float right = x + 1 < u.w ? u.at(x + 1, y) : 0.0f;
            next.at(x, y) = kDecay * (0.25f * (((up + down) + left) + right));
        }
}
}

Grid cpuOne(const Grid& sources, int sweeps) {
    Grid u = initial(sources), next(sources.w, sources.h);
    for (int s = 0; s < sweeps; ++s) {
        sweepRows(sources, u, next, 0, u.h);
        std::swap(u, next);
    }
    return u;
}

Grid cpuMulti(const Grid& sources, int sweeps, int threads) {
    Grid a = initial(sources), b(sources.w, sources.h);
    const int t = std::max(1, threads);
    Barrier barrier(static_cast<std::size_t>(t));
    std::vector<std::thread> workers;
    for (int i = 0; i < t; ++i) {
        const int y0 = sources.h * i / t, y1 = sources.h * (i + 1) / t;
        workers.emplace_back([&, y0, y1] {
            for (int s = 0; s < sweeps; ++s) {
                sweepRows(sources, s % 2 == 0 ? a : b, s % 2 == 0 ? b : a, y0, y1);
                barrier.arriveAndWait();
            }
        });
    }
    for (std::thread& w : workers) w.join();
    return sweeps % 2 == 0 ? a : b;
}

#ifndef EQ_USE_CUDA
GpuTiming gpu(const Grid&, int, Grid&) { throw std::runtime_error("built without CUDA"); }
#endif
}
