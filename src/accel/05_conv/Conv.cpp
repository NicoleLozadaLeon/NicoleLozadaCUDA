#include "Conv.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "Parallel.h"

namespace conv {
namespace {
void rows(const Grid& in, const std::vector<float>& filter, int r, int y0, int y1, Grid& out) {
    const int side = 2 * r + 1;
    for (int y = y0; y < y1; ++y)
        for (int x = 0; x < in.w; ++x) {
            float sum = 0.0f;
            for (int fy = -r; fy <= r; ++fy) {
                const int yy = y + fy;
                if (yy < 0 || yy >= in.h) continue;
                for (int fx = -r; fx <= r; ++fx) {
                    const int xx = x + fx;
                    if (xx < 0 || xx >= in.w) continue;
                    sum += filter[static_cast<std::size_t>((fy + r) * side + fx + r)] * in.at(xx, yy);
                }
            }
            out.at(x, y) = sum;
        }
}
}

std::vector<float> gaussian(int radius) {
    if (radius < 1 || radius > kMaxRadius) throw std::runtime_error("filter radius must be 1..6");
    const int side = 2 * radius + 1;
    const double sigma = radius / 2.0 + 0.5;
    std::vector<float> f(static_cast<std::size_t>(side * side));
    double total = 0.0;
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x) total += std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
    for (int y = -radius; y <= radius; ++y)
        for (int x = -radius; x <= radius; ++x)
            f[static_cast<std::size_t>((y + radius) * side + x + radius)] =
                static_cast<float>(std::exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / total);
    return f;
}

Grid cpuOne(const Grid& in, const std::vector<float>& filter, int radius) {
    Grid out(in.w, in.h);
    rows(in, filter, radius, 0, in.h, out);
    return out;
}

Grid cpuMulti(const Grid& in, const std::vector<float>& filter, int radius, int threads) {
    Grid out(in.w, in.h);
    parallelFor(static_cast<std::size_t>(in.h), threads, [&](std::size_t y0, std::size_t y1, std::size_t) {
        rows(in, filter, radius, static_cast<int>(y0), static_cast<int>(y1), out);
    });
    return out;
}

#ifndef EQ_USE_CUDA
GpuTiming gpu(const Grid&, const std::vector<float>&, int, Grid&) { throw std::runtime_error("built without CUDA"); }
#endif
}
