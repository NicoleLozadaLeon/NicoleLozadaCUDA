#include <string>
#include "Bench.h"
#include "Conv.h"
#include "Synthetic.h"

namespace bench {
void convCases(Context& ctx, const std::string& app, int radius, int hotspots, unsigned seed) {
    const std::vector<double> cellsM = {20.0, 10.0, 5.0, 2.0, 1.0};
    const std::vector<float> filter = conv::gaussian(radius);
    for (std::size_t i = 0; i < cellsM.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, cellsM.size())) continue;
        const MapGrid map(ctx.cfg.map.bounds, cellsM[i]);
        const Grid in = scatter(map, map.width() * map.height() / 50 + 1, hotspots, seed + static_cast<unsigned>(i));
        const Grid ref = conv::cpuOne(in, filter, radius);
        auto close = [&ref](const Grid& out) { return maxRelativeError(out, ref) <= conv::kTolerance; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return conv::cpuOne(in, filter, radius); },
               [&](int t) { return conv::cpuMulti(in, filter, radius, t); }, close);
        addGpu<Grid>(variants, ctx, "gpu", [&](Grid& out) { return conv::gpu(in, filter, radius, out); }, close);
        runCase(ctx, app, "05_conv", gridLabel(cellsM[i], in), static_cast<double>(in.v.size()), variants);
    }
}

namespace {
void convApp(Context& ctx) { convCases(ctx, "conv", 2, 0, 8000u); }
const Registrar kRegistered(6, "conv", convApp);
}
}
