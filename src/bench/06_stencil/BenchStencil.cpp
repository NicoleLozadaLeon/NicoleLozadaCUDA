#include <string>
#include "Bench.h"
#include "Stencil.h"
#include "Synthetic.h"

namespace bench {
namespace {
constexpr int kSweeps = 100;

void stencilApp(Context& ctx) {
    const std::vector<double> cellsM = {20.0, 10.0, 5.0, 2.0, 1.0};
    for (std::size_t i = 0; i < cellsM.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, cellsM.size())) continue;
        const MapGrid map(ctx.cfg.map.bounds, cellsM[i]);
        const Grid sources = scatter(map, map.width() * map.height() / 100 + 1, 0, 10000u + static_cast<unsigned>(i));
        const Grid ref = stencil::cpuOne(sources, kSweeps);
        auto close = [&ref](const Grid& out) { return maxRelativeError(out, ref) <= stencil::kTolerance; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return stencil::cpuOne(sources, kSweeps); },
               [&](int t) { return stencil::cpuMulti(sources, kSweeps, t); }, close);
        addGpu<Grid>(variants, ctx, "gpu", [&](Grid& out) { return stencil::gpu(sources, kSweeps, out); }, close);
        runCase(ctx, "stencil", "06_stencil", gridLabel(cellsM[i], sources), static_cast<double>(sources.v.size()),
                variants);
    }
}
const Registrar kRegistered(7, "stencil", stencilApp);
}
}
