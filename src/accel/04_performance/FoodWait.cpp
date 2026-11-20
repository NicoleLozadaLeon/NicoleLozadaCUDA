#include "FoodWait.h"
#include <stdexcept>
#include "Parallel.h"

namespace foodwait {
namespace {
int waitingMs(long long readyAtMs, bool pickedUp, long long nowMs) {
    const long long waited = nowMs - readyAtMs;
    if (pickedUp || waited <= 0) return 0;
    return static_cast<int>(waited > 0x7fffffff ? 0x7fffffff : waited);
}
}

Items toSoA(const std::vector<ItemAoS>& items) {
    Items o;
    o.readyAtMs.reserve(items.size());
    o.pickedUp.reserve(items.size());
    for (const ItemAoS& it : items) {
        o.readyAtMs.push_back(it.readyAtMs);
        o.pickedUp.push_back(it.pickedUp ? 1 : 0);
    }
    return o;
}

std::vector<int> cpuOne(const Items& items, long long nowMs) {
    std::vector<int> out(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) out[i] = waitingMs(items.readyAtMs[i], items.pickedUp[i] != 0, nowMs);
    return out;
}

std::vector<int> cpuMulti(const Items& items, long long nowMs, int threads) {
    std::vector<int> out(items.size());
    parallelFor(items.size(), threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t i = begin; i < end; ++i) out[i] = waitingMs(items.readyAtMs[i], items.pickedUp[i] != 0, nowMs);
    });
    return out;
}

#ifndef EQ_USE_CUDA
GpuTiming gpuSoA(const Items&, long long, std::vector<int>&) { throw std::runtime_error("built without CUDA"); }
GpuTiming gpuAoS(const std::vector<ItemAoS>&, long long, std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
#endif
}
