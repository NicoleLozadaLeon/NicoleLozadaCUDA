#include <string>
#include "Bench.h"
#include "Histogram.h"
#include "Synthetic.h"

namespace bench {
namespace {
const std::vector<long long> kEvents = {10000, 100000, 1000000, 10000000, 32000000};

void histogramApp(Context& ctx) {
    for (std::size_t i = 0; i < kEvents.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kEvents.size())) continue;
        const std::vector<int> ms = deliveryTimes(kEvents[i], 4000u + static_cast<unsigned>(i));
        const std::vector<unsigned> ref = hist::byWidthOne(ms, 60000, 120);
        auto same = [&ref](const std::vector<unsigned>& out) { return out == ref; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return hist::byWidthOne(ms, 60000, 120); },
               [&](int t) { return hist::byWidthMulti(ms, 60000, 120, t); }, same);
        addGpu<std::vector<unsigned>>(variants, ctx, "gpu",
                                      [&](std::vector<unsigned>& out) { return hist::byWidthGpu(ms, 60000, 120, out); },
                                      same);
        runCase(ctx, "hist", "07_histogram", std::to_string(kEvents[i]), static_cast<double>(kEvents[i]), variants);
    }
}
const Registrar kRegistered(4, "hist", histogramApp);
}
}
