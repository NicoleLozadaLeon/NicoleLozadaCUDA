#include <random>
#include <string>
#include <vector>
#include "Bench.h"
#include "FoodWait.h"

namespace bench {
namespace {
// 6000 = the bags of the city configuration: 2000 couriers x a bag of 3
const std::vector<int> kSizes = {1000, 6000, 100000, 1000000, 4000000};
constexpr long long kNowMs = 2LL * 3600 * 1000;  // two simulated hours into the run

std::vector<foodwait::ItemAoS> makeItems(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> node(0, 3599), pick(0, 99);
    std::uniform_int_distribution<long long> ready(kNowMs - 600000, kNowMs + 600000);  // +- 10 min around now
    std::vector<foodwait::ItemAoS> items(static_cast<std::size_t>(n));
    for (foodwait::ItemAoS& it : items) {
        it.restaurantNode = node(rng);
        it.deliveryNode = node(rng);
        it.readyAtMs = ready(rng);
        it.pickedUp = pick(rng) < 30;  // 30 % already picked up
    }
    return items;
}

void foodWaitApp(Context& ctx) {
    for (std::size_t i = 0; i < kSizes.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kSizes.size())) continue;
        const std::vector<foodwait::ItemAoS> aos = makeItems(kSizes[i], 14000u + static_cast<unsigned>(i));
        const foodwait::Items soa = foodwait::toSoA(aos);
        const std::vector<int> ref = foodwait::cpuOne(soa, kNowMs);
        auto same = [&ref](const std::vector<int>& out) { return out == ref; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return foodwait::cpuOne(soa, kNowMs); },
               [&](int t) { return foodwait::cpuMulti(soa, kNowMs, t); }, same);
        addGpu<std::vector<int>>(variants, ctx, "gpu",
                                 [&](std::vector<int>& out) { return foodwait::gpuSoA(soa, kNowMs, out); }, same);
        addGpu<std::vector<int>>(variants, ctx, "gpu_aos",
                                 [&](std::vector<int>& out) { return foodwait::gpuAoS(aos, kNowMs, out); }, same);
        runCase(ctx, "foodwait", "04_performance", std::to_string(kSizes[i]), static_cast<double>(kSizes[i]), variants);
    }
}
const Registrar kRegistered(9, "foodwait", foodWaitApp);
}
}
