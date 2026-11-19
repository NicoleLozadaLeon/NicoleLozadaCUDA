#include "Eta.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "Floyd.h"
#include "Parallel.h"

namespace eta {
namespace {
int etaOf(const std::vector<int>& dist, const Params& p, const Query& q, int freeNode, long long freeAtMs) {
    const std::size_t n = static_cast<std::size_t>(p.n);
    const int toRestaurant = dist[static_cast<std::size_t>(freeNode) * n + static_cast<std::size_t>(q.restaurantNode)];
    const int toCustomer = dist[static_cast<std::size_t>(q.restaurantNode) * n + static_cast<std::size_t>(q.deliveryNode)];
    if (toRestaurant >= floyd::kInf || toCustomer >= floyd::kInf) return kNoEta;
    long long t = std::max(p.nowMs, freeAtMs) + travelMs(toRestaurant, p.metersPerHour);
    t = std::max(t, q.readyAtMs) + p.handoverMs;
    t += travelMs(toCustomer, p.metersPerHour);
    return static_cast<int>(std::clamp(t - p.nowMs, 0LL, static_cast<long long>(kNoEta - 1)));
}

void rows(const std::vector<int>& dist, const Params& p, const std::vector<Query>& orders, const Couriers& couriers,
          std::size_t begin, std::size_t end, std::vector<int>& out) {
    const std::size_t k = couriers.size();
    for (std::size_t i = begin; i < end; ++i)
        for (std::size_t j = 0; j < k; ++j)
            out[i * k + j] = etaOf(dist, p, orders[i], couriers.freeNode[j], couriers.freeAtMs[j]);
}
}

std::vector<CourierAoS> toAoS(const Couriers& c) {
    std::vector<CourierAoS> out(c.size());
    for (std::size_t j = 0; j < c.size(); ++j)
        out[j] = {static_cast<int>(j), 0, 1, c.freeNode[j], c.freeNode[j], 0, c.freeAtMs[j]};
    return out;
}

std::vector<int> cpuOne(const std::vector<int>& dist, const Params& p, const std::vector<Query>& orders,
                        const Couriers& couriers) {
    std::vector<int> out(orders.size() * couriers.size());
    rows(dist, p, orders, couriers, 0, orders.size(), out);
    return out;
}

std::vector<int> cpuMulti(const std::vector<int>& dist, const Params& p, const std::vector<Query>& orders,
                          const Couriers& couriers, int threads) {
    std::vector<int> out(orders.size() * couriers.size());
    parallelFor(orders.size(), threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        rows(dist, p, orders, couriers, begin, end, out);
    });
    return out;
}

#ifndef EQ_USE_CUDA
DeviceTable::DeviceTable(const std::vector<int>&, int) { throw std::runtime_error("built without CUDA"); }
DeviceTable::~DeviceTable() = default;
GpuTiming gpuSoA(const DeviceTable&, const Params&, const std::vector<Query>&, const Couriers&, std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
GpuTiming gpuAoS(const DeviceTable&, const Params&, const std::vector<Query>&, const std::vector<CourierAoS>&,
                 std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
#endif
}
