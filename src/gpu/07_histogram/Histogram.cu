#include "Histogram.h"
#include <chrono>
#include "HistKernels.h"

namespace hist {
namespace {
struct WidthBin {
    const int* values;
    int width;
    int last;
    __device__ int operator()(long long i) const {
        const int b = values[i] / width;
        return b < last ? b : last;
    }
};
}

void launchByWidth(const int* values, long long n, int width, int bins, unsigned* out) {
    launchPrivatized(n, bins, WidthBin{values, width, bins - 1}, out);
}

GpuTiming byWidthGpu(const std::vector<int>& values, int width, int bins, std::vector<unsigned>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    DeviceBuffer<int> dValues;
    DeviceBuffer<unsigned> dOut(static_cast<std::size_t>(bins));
    dValues.upload(values.data(), values.size());
    dOut.zero();
    KernelTimer timer;
    timer.start();
    launchByWidth(dValues.data(), static_cast<long long>(values.size()), width, bins, dOut.data());
    timing.kernelMs = timer.stop();
    out.resize(static_cast<std::size_t>(bins));
    dOut.download(out.data(), out.size());
    timing.e2eMs = msSince(t0);
    return timing;
}
}
