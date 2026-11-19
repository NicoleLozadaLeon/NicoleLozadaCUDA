#pragma once
#include <cstddef>
#include <vector>
#include "Gpu.h"

namespace eta {
constexpr int kNoEta = 0x7fffffff;

struct Query {
    int restaurantNode;
    int deliveryNode;
    long long readyAtMs;
};

struct Params {
    int n;
    long long nowMs;
    long long metersPerHour;
    long long handoverMs;
};

struct CourierAoS {
    int index;
    int state;
    int available;
    int legFrom;
    int freeNode;
    int bagSize;
    long long freeAtMs;
};

struct Couriers {
    std::vector<int> freeNode;
    std::vector<long long> freeAtMs;
    std::size_t size() const { return freeNode.size(); }
};

std::vector<CourierAoS> toAoS(const Couriers& c);

inline long long travelMs(int dm, long long metersPerHour) {
    return static_cast<long long>(dm) * 360000LL / metersPerHour;
}

std::vector<int> cpuOne(const std::vector<int>& dist, const Params& p, const std::vector<Query>& orders,
                        const Couriers& couriers);
std::vector<int> cpuMulti(const std::vector<int>& dist, const Params& p, const std::vector<Query>& orders,
                          const Couriers& couriers, int threads);

class DeviceTable {
public:
    DeviceTable(const std::vector<int>& dist, int n);
    ~DeviceTable();
    DeviceTable(const DeviceTable&) = delete;
    DeviceTable& operator=(const DeviceTable&) = delete;
    const int* data() const { return data_; }
    int n() const { return n_; }

private:
    int* data_ = nullptr;
    int n_ = 0;
};

GpuTiming gpuSoA(const DeviceTable& table, const Params& p, const std::vector<Query>& orders,
                 const Couriers& couriers, std::vector<int>& out);
GpuTiming gpuAoS(const DeviceTable& table, const Params& p, const std::vector<Query>& orders,
                 const std::vector<CourierAoS>& couriers, std::vector<int>& out);
}
