#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include "Argmin.h"
#include "Bench.h"

namespace bench {
namespace {
const std::vector<std::pair<int, int>> kSizes = {{10, 50}, {100, 500}, {500, 2000}, {1000, 5000}, {2000, 5000}};

void argminApp(Context& ctx) {
    for (std::size_t i = 0; i < kSizes.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kSizes.size())) continue;
        const auto [m, k] = kSizes[i];
        std::mt19937 rng(3000u + static_cast<unsigned>(i));
        std::uniform_int_distribution<int> etaMs(0, 2000);
        std::bernoulli_distribution unreachable(0.02);
        std::vector<int> etas(static_cast<std::size_t>(m) * k);
        for (int& e : etas) e = unreachable(rng) ? eta::kNoEta : etaMs(rng);
        std::vector<int> ids(static_cast<std::size_t>(k));
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        const std::vector<argmin::Key> ref = argmin::cpuOne(etas, m, k, ids);
        auto same = [&ref](const std::vector<argmin::Key>& out) { return out == ref; };
        std::vector<Variant> variants;
        addCpu(variants, ctx.opt, [&] { return argmin::cpuOne(etas, m, k, ids); },
               [&](int t) { return argmin::cpuMulti(etas, m, k, ids, t); }, same);
        addGpu<std::vector<argmin::Key>>(variants, ctx, "gpu",
                                         [&](std::vector<argmin::Key>& out) { return argmin::gpu(etas, m, k, ids, out); },
                                         same);
        runCase(ctx, "argmin", "08_reduction", std::to_string(m) + "x" + std::to_string(k),
                static_cast<double>(m) * k, variants);
    }
}
const Registrar kRegistered(3, "argmin", argminApp);
}
}
