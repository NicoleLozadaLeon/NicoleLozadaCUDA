#include <string>
#include "Bench.h"
#include "Scan.h"
#include "Synthetic.h"

namespace bench {
namespace {
const std::vector<long long> kOrders = {1000, 10000, 100000, 1000000, 4000000};
constexpr int kRestaurants = 64;

void groupApp(Context& ctx) {
    for (std::size_t i = 0; i < kOrders.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kOrders.size())) continue;
        const std::vector<int> keys = restaurantKeys(kOrders[i], kRestaurants, 5000u + static_cast<unsigned>(i));
        const scan::Grouping ref = scan::groupOne(keys, kRestaurants);
        auto same = [&ref](const scan::Grouping& out) { return out == ref; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return scan::groupOne(keys, kRestaurants); },
               [&](int t) { return scan::groupMulti(keys, kRestaurants, t); }, same);
        addGpu<scan::Grouping>(variants, ctx, "gpu",
                               [&](scan::Grouping& out) { return scan::groupGpu(keys, kRestaurants, out); }, same);
        runCase(ctx, "group", "09_scan", std::to_string(kOrders[i]), static_cast<double>(kOrders[i]), variants);
    }
}
const Registrar kRegistered(5, "group", groupApp);
}
}
