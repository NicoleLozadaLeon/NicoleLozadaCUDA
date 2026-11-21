#pragma once
#include <cstddef>
#include <vector>
#include "Gpu.h"

// Food waiting: how long each order in a courier's bag has been ready without being picked up
// (concept 04, second application). Times are whole milliseconds, so every result is exact.
namespace foodwait {

// Array of structures: the same four fields as Courier::View::Item, the record the program already
// keeps for every order in a bag. 24 bytes, of which the check reads only readyAtMs and pickedUp.
struct ItemAoS {
    int restaurantNode;
    int deliveryNode;
    long long readyAtMs;
    bool pickedUp;
};

// Structure of arrays: only the two fields the check reads, each in its own array.
struct Items {
    std::vector<long long> readyAtMs;
    std::vector<unsigned char> pickedUp;
    std::size_t size() const { return readyAtMs.size(); }
};

Items toSoA(const std::vector<ItemAoS>& items);

// out[i] = 0 if order i is already picked up or not ready yet, else nowMs - readyAtMs (at most INT_MAX).
std::vector<int> cpuOne(const Items& items, long long nowMs);
std::vector<int> cpuMulti(const Items& items, long long nowMs, int threads);
GpuTiming gpuSoA(const Items& items, long long nowMs, std::vector<int>& out);
GpuTiming gpuAoS(const std::vector<ItemAoS>& items, long long nowMs, std::vector<int>& out);
}
