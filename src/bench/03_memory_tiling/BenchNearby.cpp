#include <memory>
#include <random>
#include <string>
#include <utility>
#include "Bench.h"
#include "MapGrid.h"
#include "Nearby.h"

namespace bench {
namespace {
// corners x couriers; 3600 x 2000 is the city configuration (city_60)
const std::vector<std::pair<int, int>> kSizes = {{100, 50}, {400, 200}, {1600, 1000}, {3600, 2000}, {10000, 5000}};

nearby::Points scatterPoints(int n, int width, int height, std::mt19937& rng) {
    std::uniform_int_distribution<int> x(0, width - 1), y(0, height - 1);
    nearby::Points p;
    for (int i = 0; i < n; ++i) {
        p.x.push_back(x(rng));
        p.y.push_back(y(rng));
    }
    return p;
}

void nearbyApp(Context& ctx) {
    const MapGrid metres(ctx.cfg.map.bounds, 1.0);  // 1 m cells: the map size in metres
    for (std::size_t i = 0; i < kSizes.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kSizes.size())) continue;
        const auto [corners, couriers] = kSizes[i];
        std::mt19937 rng(12000u + static_cast<unsigned>(i));
        const nearby::Points c = scatterPoints(corners, metres.width(), metres.height(), rng);
        const nearby::Points q = scatterPoints(couriers, metres.width(), metres.height(), rng);
        const std::vector<int> ref = nearby::cpuOne(c, q, nearby::kRadiusM);
        auto same = [&ref](const std::vector<int>& out) { return out == ref; };
        // The corners go to the GPU once, outside the timing, as in the program (the map never changes).
        std::unique_ptr<nearby::DeviceCorners> dc;
        if (ctx.gpu) dc = std::make_unique<nearby::DeviceCorners>(c);
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return nearby::cpuOne(c, q, nearby::kRadiusM); },
               [&](int t) { return nearby::cpuMulti(c, q, nearby::kRadiusM, t); }, same);
        addGpu<std::vector<int>>(variants, ctx, "gpu",
                                 [&](std::vector<int>& out) { return nearby::gpuTiled(*dc, q, nearby::kRadiusM, out); },
                                 same);
        addGpu<std::vector<int>>(variants, ctx, "gpu_naive",
                                 [&](std::vector<int>& out) { return nearby::gpuNaive(*dc, q, nearby::kRadiusM, out); },
                                 same);
        addGpu<std::vector<int>>(
            variants, ctx, "gpu_upload",
            [&](std::vector<int>& out) { return nearby::gpuUploadAll(c, q, nearby::kRadiusM, out); }, same);
        runCase(ctx, "near", "03_memory_tiling", std::to_string(corners) + "x" + std::to_string(couriers),
                static_cast<double>(corners) * couriers, variants);
    }
}
const Registrar kRegistered(8, "near", nearbyApp);
}
}
