#include <random>
#include <string>
#include <vector>
#include "Bench.h"
#include "FleetSummary.h"

namespace bench {
namespace {
// 2000 = the fleet of the city configuration (city_60)
const std::vector<int> kSizes = {100, 2000, 10000, 100000, 1000000};
constexpr long long kNowMs = 2LL * 3600 * 1000;  // two simulated hours into the run
constexpr int kBagCapacity = 3;

// Synthetic fleet with a fixed seed. Three couriers get exactly the same, largest, leg end: the argmax must
// pick the lowest of the three. Many couriers share the fullest bag (3 items): a tie on every run.
fleetsum::FleetData makeFleet(int n, unsigned seed, unsigned& lowestTied) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, 99);
    std::uniform_int_distribution<long long> legEnd(0, kNowMs);
    fleetsum::FleetData f;
    for (int i = 0; i < n; ++i) {
        const int p = pick(rng);
        const unsigned char state = p < 40 ? fleetsum::kIdle : p < 70 ? fleetsum::kToRestaurant : p < 95 ? fleetsum::kCarrying : fleetsum::kBroken;
        unsigned char items = 0, picked = 0;
        if (state == fleetsum::kToRestaurant) {
            items = static_cast<unsigned char>(1 + rng() % kBagCapacity);
            picked = static_cast<unsigned char>(rng() % items);          // some orders still to pick up
        } else if (state == fleetsum::kCarrying) {
            items = static_cast<unsigned char>(1 + rng() % kBagCapacity);
            picked = static_cast<unsigned char>(1 + rng() % items);      // at least one is on board
        }
        f.state.push_back(state);
        f.items.push_back(items);
        f.picked.push_back(picked);
        f.available.push_back(state != fleetsum::kBroken && items < kBagCapacity ? 1 : 0);
        f.legEndMs.push_back(legEnd(rng));
    }
    const unsigned tied[3] = {static_cast<unsigned>(n / 7), static_cast<unsigned>(n / 3), static_cast<unsigned>(n - 1)};
    for (unsigned i : tied) f.legEndMs[i] = 3 * kNowMs;
    lowestTied = tied[0];
    return f;
}

void fleetSummaryApp(Context& ctx) {
    for (std::size_t i = 0; i < kSizes.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kSizes.size())) continue;
        unsigned lowestTied = 0;
        const fleetsum::FleetData fleet = makeFleet(kSizes[i], 15000u + static_cast<unsigned>(i), lowestTied);
        const fleetsum::Summary ref = fleetsum::cpuOne(fleet);
        const unsigned n = static_cast<unsigned>(kSizes[i]);
        // exact, the four states add up to the whole fleet, and the tie goes to the lowest index
        auto same = [&](const fleetsum::Summary& out) {
            return fleetsum::equal(out, ref) && fleetsum::couriersCounted(out) == n && out.latestEndIdx == lowestTied;
        };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return fleetsum::cpuOne(fleet); },
               [&](int t) { return fleetsum::cpuMulti(fleet, t); }, same);
        addGpu<fleetsum::Summary>(variants, ctx, "gpu",
                                  [&](fleetsum::Summary& out) { return fleetsum::gpuBranchFree(fleet, out); }, same);
        addGpu<fleetsum::Summary>(variants, ctx, "gpu_branchy",
                                  [&](fleetsum::Summary& out) { return fleetsum::gpuBranchy(fleet, out); }, same);
        runCase(ctx, "fleetsum", "08_reduction", std::to_string(kSizes[i]), static_cast<double>(kSizes[i]), variants);
    }
}
const Registrar kRegistered(10, "fleetsum", fleetSummaryApp);
}
}
