#pragma once
#include <vector>
#include "Gpu.h"

// How many couriers are within a radius of every street corner (concept 03, second application).
// Positions are whole metres from the north-west corner of the map, so every count is exact.
namespace nearby {
constexpr int kRadiusM = 500;

struct Points {  // structure of arrays: x[i], y[i] is point i
    std::vector<int> x, y;
};

std::vector<int> cpuOne(const Points& corners, const Points& couriers, int radiusM);
std::vector<int> cpuMulti(const Points& corners, const Points& couriers, int radiusM, int threads);

// The street corners never move, so they are uploaded once and stay on the GPU, together with the
// output buffer. Each call only uploads the couriers.
class DeviceCorners {
public:
    explicit DeviceCorners(const Points& corners);
    ~DeviceCorners();
    DeviceCorners(const DeviceCorners&) = delete;
    DeviceCorners& operator=(const DeviceCorners&) = delete;
    const int* x() const { return data_; }
    const int* y() const { return data_ + n_; }
    int* out() const { return out_; }
    int n() const { return n_; }

private:
    int* data_ = nullptr;
    int* out_ = nullptr;
    int n_ = 0;
};

GpuTiming gpuTiled(const DeviceCorners& corners, const Points& couriers, int radiusM, std::vector<int>& out);
GpuTiming gpuNaive(const DeviceCorners& corners, const Points& couriers, int radiusM, std::vector<int>& out);
// Worst case for comparison: corners and couriers uploaded on every call.
GpuTiming gpuUploadAll(const Points& corners, const Points& couriers, int radiusM, std::vector<int>& out);
}
