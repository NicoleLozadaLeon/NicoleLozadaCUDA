#include "FleetSummary.h"
#include <stdexcept>
#include "Parallel.h"

namespace fleetsum {
namespace {
bool beats(unsigned long long va, unsigned ia, unsigned long long vb, unsigned ib) {
    return va > vb || (va == vb && ia < ib);
}

void add(Summary& s, unsigned i, const FleetData& f) {
    ++s.state[f.state[i]];
    s.available += f.available[i];
    s.items += f.items[i];
    s.picked += f.picked[i];
    if (beats(f.items[i], i, s.mostItems, s.mostItemsIdx)) {
        s.mostItems = f.items[i];
        s.mostItemsIdx = i;
    }
    const auto end = static_cast<unsigned long long>(f.legEndMs[i]);
    if (beats(end, i, s.latestEnd, s.latestEndIdx)) {
        s.latestEnd = end;
        s.latestEndIdx = i;
    }
}

Summary combine(const Summary& a, const Summary& b) {
    Summary r = a;
    for (int k = 0; k < 4; ++k) r.state[k] += b.state[k];
    r.available += b.available;
    r.items += b.items;
    r.picked += b.picked;
    if (beats(b.mostItems, b.mostItemsIdx, a.mostItems, a.mostItemsIdx)) {
        r.mostItems = b.mostItems;
        r.mostItemsIdx = b.mostItemsIdx;
    }
    if (beats(b.latestEnd, b.latestEndIdx, a.latestEnd, a.latestEndIdx)) {
        r.latestEnd = b.latestEnd;
        r.latestEndIdx = b.latestEndIdx;
    }
    return r;
}
}

Summary identity() {
    Summary s{};
    s.mostItemsIdx = kNoIndex;
    s.latestEndIdx = kNoIndex;
    return s;
}

bool equal(const Summary& a, const Summary& b) {
    for (int k = 0; k < 4; ++k)
        if (a.state[k] != b.state[k]) return false;
    return a.available == b.available && a.items == b.items && a.picked == b.picked && a.mostItems == b.mostItems &&
           a.mostItemsIdx == b.mostItemsIdx && a.latestEnd == b.latestEnd && a.latestEndIdx == b.latestEndIdx;
}

unsigned couriersCounted(const Summary& s) { return s.state[0] + s.state[1] + s.state[2] + s.state[3]; }

// Reference: one pass over the couriers.
Summary cpuOne(const FleetData& fleet) {
    Summary s = identity();
    for (unsigned i = 0; i < fleet.size(); ++i) add(s, i, fleet);
    return s;
}

// Each thread folds its own range into a local summary; the partial summaries are combined in thread
// order. combine() is associative and commutative and has an identity, so the result is the same.
Summary cpuMulti(const FleetData& fleet, int threads) {
    std::vector<Summary> parts(static_cast<std::size_t>(std::max(1, threads)), identity());
    parallelFor(fleet.size(), threads, [&](std::size_t begin, std::size_t end, std::size_t id) {
        Summary s = identity();
        for (std::size_t i = begin; i < end; ++i) add(s, static_cast<unsigned>(i), fleet);
        parts[id] = s;
    });
    Summary total = identity();
    for (const Summary& p : parts) total = combine(total, p);
    return total;
}

#ifndef EQ_USE_CUDA
GpuTiming gpuBranchFree(const FleetData&, Summary&) { throw std::runtime_error("built without CUDA"); }
GpuTiming gpuBranchy(const FleetData&, Summary&) { throw std::runtime_error("built without CUDA"); }
#endif
}
