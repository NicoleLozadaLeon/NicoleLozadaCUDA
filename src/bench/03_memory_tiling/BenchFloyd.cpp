#include <string>
#include "Bench.h"
#include "Floyd.h"

namespace bench {
namespace {
void floydApp(Context& ctx) {
    const std::vector<int> sides = {10, 17, 32, 45, 70};
    for (std::size_t i = 0; i < sides.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, sides.size())) continue;
        const floyd::Network net = floyd::gridNetwork(sides[i], 1000u + static_cast<unsigned>(i));
        const std::vector<int> ref = floyd::dijkstraOne(net);
        auto same = [&ref](const std::vector<int>& d) { return d == ref; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return floyd::dijkstraOne(net); },
               [&](int t) { return floyd::dijkstraMulti(net, t); }, same);
        addGpu<std::vector<int>>(variants, ctx, "gpu", [&](std::vector<int>& out) { return floyd::floydGpu(net, out); },
                                 same);
        runCase(ctx, "fw", "03_memory_tiling", "n=" + std::to_string(net.n), net.n, variants);
    }
}
const Registrar kRegistered(1, "fw", floydApp);
}
}