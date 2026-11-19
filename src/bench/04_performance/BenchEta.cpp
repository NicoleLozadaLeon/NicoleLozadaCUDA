#include <memory>
#include <random>
#include <string>
#include <utility>
#include "Bench.h"
#include "Eta.h"
#include "Floyd.h"
#include "Parallel.h"

namespace bench {
namespace {
const std::vector<std::pair<int, int>> kSizes = {{10, 50}, {100, 500}, {500, 2000}, {1000, 5000}, {2000, 5000}};

struct Batch {
    std::vector<eta::Query> orders;
    eta::Couriers couriers;
};

Batch makeBatch(int m, int k, int n, long long now, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> node(0, n - 1);
    std::uniform_int_distribution<long long> ready(0, 600000), busy(0, 1200000);
    Batch b;
    for (int i = 0; i < m; ++i) b.orders.push_back({node(rng), node(rng), now + ready(rng)});
    for (int j = 0; j < k; ++j) {
        b.couriers.freeNode.push_back(node(rng));
        b.couriers.freeAtMs.push_back(now + busy(rng) - 300000);
    }
    return b;
}

void etaApp(Context& ctx) {
    const floyd::Network net = floyd::gridNetwork(32, 77u);
    const std::vector<int> dist = floyd::dijkstraMulti(net, hardwareThreads());
    const eta::Params p{net.n, 1000000, 30000, 30000};
    std::unique_ptr<eta::DeviceTable> table;
    if (ctx.gpu) table = std::make_unique<eta::DeviceTable>(dist, net.n);
    for (std::size_t i = 0; i < kSizes.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kSizes.size())) continue;
        const auto [m, k] = kSizes[i];
        const Batch b = makeBatch(m, k, net.n, p.nowMs, 2000u + static_cast<unsigned>(i));
        const std::vector<eta::CourierAoS> aos = eta::toAoS(b.couriers);
        const std::vector<int> ref = eta::cpuOne(dist, p, b.orders, b.couriers);
        auto same = [&ref](const std::vector<int>& out) { return out == ref; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return eta::cpuOne(dist, p, b.orders, b.couriers); },
               [&](int t) { return eta::cpuMulti(dist, p, b.orders, b.couriers, t); }, same);
        addGpu<std::vector<int>>(variants, ctx, "gpu",
                                 [&](std::vector<int>& out) { return eta::gpuSoA(*table, p, b.orders, b.couriers, out); },
                                 same);
        addGpu<std::vector<int>>(variants, ctx, "gpu_aos",
                                 [&](std::vector<int>& out) { return eta::gpuAoS(*table, p, b.orders, aos, out); }, same);
        runCase(ctx, "eta", "04_performance", std::to_string(m) + "x" + std::to_string(k), static_cast<double>(m) * k,
                variants);
    }
}
const Registrar kRegistered(2, "eta", etaApp);
}
}
