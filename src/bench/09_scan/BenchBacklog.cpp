#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "Backlog.h"
#include "Bench.h"

namespace bench {
namespace {
const std::vector<int> kOrders = {500, 5000, 50000, 500000, 4000000};  // orders created in the day
constexpr long long kHourMs = 3600LL * 1000;
constexpr long long kDayMs = 24 * kHourMs;
constexpr long long kAcceptTimeoutMs = 600000;  // dispatch.acceptTimeoutMs of the configuration: 10 minutes
constexpr double kAssignedShare = 0.385;        // of the orders that were resolved in the log of city_60 (11 730 / 30 484)
constexpr double kMeanAssignDelayMs = 3000.0;
// Orders per hour of the day, the profile of bikeshare_rentals_in_progress.cu: two rush hours.
constexpr int kHourWeights[24] = {1, 1, 1, 1, 1, 2, 5, 12, 20, 12, 6, 5, 7, 7, 5, 5, 7, 13, 20, 13, 7, 4, 3, 2};

// Synthetic day with a fixed seed. Every order gives +1 when it is created. Then, as in the log of city_60,
// it is assigned a few seconds later (38.5 %) or rejected exactly at the accept timeout. An order that is still
// open when the day ends gives no -1. At equal times the +1 goes first, as the course puts Undock before Dock.
backlog::EventStream makeDay(int orders, unsigned seed) {
    std::mt19937_64 rng(seed);
    const int total = std::accumulate(kHourWeights, kHourWeights + 24, 0);
    std::uniform_int_distribution<int> pickHour(0, total - 1);
    std::uniform_int_distribution<long long> inHour(0, kHourMs - 1), jitter(0, 99);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::exponential_distribution<double> delay(1.0 / kMeanAssignDelayMs);
    std::vector<std::pair<long long, int>> events;
    events.reserve(static_cast<std::size_t>(orders) * 2);
    for (int i = 0; i < orders; ++i) {
        int pick = pickHour(rng), hour = 0;
        while (pick >= kHourWeights[hour]) pick -= kHourWeights[hour++];
        const long long created = hour * kHourMs + inHour(rng);
        events.push_back({created, +1});
        const long long resolved = unit(rng) < kAssignedShare ? created + static_cast<long long>(delay(rng))
                                                              : created + kAcceptTimeoutMs + jitter(rng);
        if (resolved < kDayMs) events.push_back({resolved, -1});
    }
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second > b.second;
    });
    backlog::EventStream s;
    s.deltas.reserve(events.size());
    s.timesMs.reserve(events.size());
    for (const auto& e : events) {
        s.timesMs.push_back(e.first);
        s.deltas.push_back(e.second);
    }
    return s;
}

void runOne(Context& ctx, const std::string& label, const backlog::EventStream& stream) {
    const backlog::Result ref = backlog::cpuOne(stream);
    const int endValue = std::accumulate(stream.deltas.begin(), stream.deltas.end(), 0);  // a reduction apart from the scan
    // exact, never negative, and it ends where the independent sum of the events says
    auto same = [&](const backlog::Result& out) {
        if (!backlog::equal(out, ref)) return false;
        if (out.waiting.empty()) return true;
        return out.waiting.back() == endValue && *std::min_element(out.waiting.begin(), out.waiting.end()) >= 0;
    };
    std::vector<Variant> variants;
    addCpu(variants, ctx.opt, [&] { return backlog::cpuOne(stream); },
           [&](int t) { return backlog::cpuMulti(stream, t); }, same);
    addGpu<backlog::Result>(variants, ctx, "gpu",
                            [&](backlog::Result& out) { return backlog::gpuScan(stream, out); }, same);
    runCase(ctx, "backlog", "09_scan", label, static_cast<double>(stream.size()), variants);
}

void backlogApp(Context& ctx) {
    for (std::size_t i = 0; i < kOrders.size(); ++i) {
        if (!sizeWanted(ctx.opt, i, kOrders.size())) continue;
        const backlog::EventStream stream = makeDay(kOrders[i], 16000u + static_cast<unsigned>(i));
        runOne(ctx, std::to_string(stream.size()) + " events", stream);
    }
    // Optional check on real data: EQ_BACKLOG_LOG=<the events.log of a run>. Only used when the variable is set.
    if (const char* path = std::getenv("EQ_BACKLOG_LOG")) {
        if (ctx.opt.sizeIndex < 0) {
            const backlog::EventStream real = backlog::loadLog(path);
            runOne(ctx, "real log " + std::to_string(real.size()) + " events", real);
            const backlog::Summary s = backlog::summarize(real, backlog::cpuOne(real));
            std::cout << "real log: peak " << s.peak << " orders waiting at t=" << s.peakAtMs << " ms, ends at "
                      << s.waitingAtEnd << "\n";
        }
    }
}
const Registrar kRegistered(11, "backlog", backlogApp);
}
}
