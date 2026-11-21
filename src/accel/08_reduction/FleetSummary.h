#pragma once
#include <cstddef>
#include <vector>
#include "Gpu.h"

// Fleet summary in one pass (concept 08, second application): counts, sums and two argmax over the
// state of every courier, as in bikeshare_daily_summary.cu and bikeshare_kpis.cu. Every field is an
// integer, so the result is exact and identical in every run.
namespace fleetsum {
constexpr unsigned kNoIndex = 0xffffffffu;

// Same order as Courier::State, so the state of a courier can be copied as it is.
enum State { kIdle = 0, kToRestaurant = 1, kCarrying = 2, kBroken = 3 };

// Structure of arrays, one entry per courier (what Courier::view() gives for every courier).
struct FleetData {
    std::vector<unsigned char> state, available, items, picked;  // items in the bag, picked up among them
    std::vector<long long> legEndMs;                             // when its current leg ends
    std::size_t size() const { return state.size(); }
};

struct Summary {
    unsigned state[4];                 // couriers in each state
    unsigned available;                // couriers that can take another order
    unsigned long long items, picked;  // orders in all bags, and how many of them are already picked up
    unsigned long long mostItems;      // argmax: the courier with the fullest bag (the lowest index wins a tie)
    unsigned mostItemsIdx;
    unsigned long long latestEnd;      // argmax: the courier whose current leg ends last
    unsigned latestEndIdx;
};

Summary identity();
bool equal(const Summary& a, const Summary& b);
unsigned couriersCounted(const Summary& s);  // sum of the four states: it must be the size of the fleet

Summary cpuOne(const FleetData& fleet);
Summary cpuMulti(const FleetData& fleet, int threads);
// Final design: the state is classified without branches (each state adds a 0 or a 1).
GpuTiming gpuBranchFree(const FleetData& fleet, Summary& out);
// For comparison: the same work with an if/switch per courier.
GpuTiming gpuBranchy(const FleetData& fleet, Summary& out);
}
