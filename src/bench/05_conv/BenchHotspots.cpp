#include <cmath>
#include <random>
#include <string>
#include <vector>
#include "Bench.h"
#include "Hotspots.h"
#include "Synthetic.h"

namespace bench {
namespace {
const std::vector<double> kCellsM = {20.0, 10.0, 5.0, 2.0, 1.0};
constexpr int kPending = 5000;  // orders still unresolved after one simulated hour of city_60 (4 968)
constexpr int kCentres = 6;
constexpr double kShareInCentres = 0.6;  // the rest of the orders are spread over the whole map
constexpr double kCentreSigmaM = 150.0;
constexpr int kWarmUpMs = 400;  // GPU clock warm-up before every timed GPU run

struct Point {
    double lat, lon;
};

// Pending orders with a fixed seed: most of them around a few centres of demand, the rest uniform on the map.
// The same points are used for every cell size, so only the grid changes between the five sizes.
std::vector<Point> makePending(const Bounds& b, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    const double midLat = (b.north + b.south) / 2.0 * 3.14159265358979323846 / 180.0;
    const double mPerLon = 111320.0 * std::cos(midLat), mPerLat = 110540.0;
    std::vector<Point> centres;
    for (int c = 0; c < kCentres; ++c)
        centres.push_back({b.south + (0.15 + 0.7 * unit(rng)) * (b.north - b.south),
                           b.west + (0.15 + 0.7 * unit(rng)) * (b.east - b.west)});
    std::vector<Point> points;
    while (static_cast<int>(points.size()) < n) {
        Point p{b.south + unit(rng) * (b.north - b.south), b.west + unit(rng) * (b.east - b.west)};
        if (unit(rng) < kShareInCentres) {
            const Point& c = centres[rng() % centres.size()];
            p.lat = c.lat + normal(rng) * kCentreSigmaM / mPerLat;
            p.lon = c.lon + normal(rng) * kCentreSigmaM / mPerLon;
            if (p.lat < b.south || p.lat > b.north || p.lon < b.west || p.lon > b.east) continue;
        }
        points.push_back(p);
    }
    return points;
}

// Before every measured GPU run: the GPU is kept busy for a while so it leaves its idle clock, and the same GPU
// variant runs once without being measured so the host memory it needs is already mapped. The times that go to
// the CSV come only from the second run, which is the one that is returned.
template <class Fn>
GpuTiming warmThenRun(Fn fn, Grid& out) {
    hotspots::gpuWarmUp(kWarmUpMs);
    Grid scratch;
    fn(scratch);
    return fn(out);
}

void hotspotsApp(Context& ctx) {
    const std::vector<Point> pending = makePending(ctx.cfg.map.bounds, kPending, 12000u);
    const std::vector<float> filter = hotspots::gaussian();
    for (std::size_t i = 0; i < kCellsM.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kCellsM.size())) continue;
        const MapGrid map(ctx.cfg.map.bounds, kCellsM[i]);
        const int w = map.width(), h = map.height();
        std::vector<int> cells;
        for (const Point& p : pending) {
            int x = 0, y = 0;
            if (map.cellOf(p.lat, p.lon, x, y)) cells.push_back(y * w + x);
        }
        const Grid ref = hotspots::cpuOne(w, h, cells, filter);
        auto close = [&ref](const Grid& out) { return maxRelativeError(out, ref) <= hotspots::kTolerance; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return hotspots::cpuOne(w, h, cells, filter); },
               [&](int t) { return hotspots::cpuMulti(w, h, cells, filter, t); }, close);
        addGpu<Grid>(variants, ctx, "gpu", [&](Grid& out) {
            return warmThenRun([&](Grid& g) { return hotspots::gpuTiled(w, h, cells, filter, g); }, out);
        }, close);
        addGpu<Grid>(variants, ctx, "gpu_direct", [&](Grid& out) {
            return warmThenRun([&](Grid& g) { return hotspots::gpuDirect(w, h, cells, filter, g); }, out);
        }, close);
        runCase(ctx, "hotspots", "05_conv", gridLabel(kCellsM[i], ref), static_cast<double>(ref.v.size()), variants);
    }
}
const Registrar kRegistered(12, "hotspots", hotspotsApp);
}
}
