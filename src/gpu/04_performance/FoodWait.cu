#include "FoodWait.h"
#include <chrono>
#include <cstddef>
#include "Cuda.h"

namespace foodwait {
namespace {
constexpr unsigned kThreads = 256;

__device__ __forceinline__ int waitingMs(long long readyAtMs, bool pickedUp, long long nowMs) {
    const long long waited = nowMs - readyAtMs;
    if (pickedUp || waited <= 0) return 0;
    return static_cast<int>(waited > 0x7fffffff ? 0x7fffffff : waited);
}

// One thread per order. Thread i reads readyAtMs[i] and pickedUp[i]: the 32 threads of a warp read
// 32 consecutive long longs (256 B) and 32 consecutive bytes (32 B), and every byte is used.
__global__ void foodWaitSoA(const long long* readyAtMs, const unsigned char* pickedUp, long long nowMs, int n,
                            int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = waitingMs(readyAtMs[i], pickedUp[i] != 0, nowMs);
}

// The same check on records: thread i reads two fields of a 24-byte record, so a warp touches
// 32 x 24 = 768 B to use 288 B.
__global__ void foodWaitAoS(const ItemAoS* items, long long nowMs, int n, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ItemAoS it = items[i];
    out[i] = waitingMs(it.readyAtMs, it.pickedUp, nowMs);
}
}

GpuTiming gpuSoA(const Items& items, long long nowMs, std::vector<int>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const std::size_t n = items.size();
    DeviceBuffer<long long> dReady;
    DeviceBuffer<unsigned char> dPicked;
    DeviceBuffer<int> dOut(n);
    dReady.upload(items.readyAtMs.data(), n);
    dPicked.upload(items.pickedUp.data(), n);
    KernelTimer timer;
    timer.start();
    if (n > 0)
        foodWaitSoA<<<blocksFor(n, kThreads), kThreads>>>(dReady.data(), dPicked.data(), nowMs, static_cast<int>(n),
                                                          dOut.data());
    timing.kernelMs = timer.stop();
    out.assign(n, 0);
    dOut.download(out.data(), n);
    timing.e2eMs = msSince(t0);
    return timing;
}

GpuTiming gpuAoS(const std::vector<ItemAoS>& items, long long nowMs, std::vector<int>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const std::size_t n = items.size();
    DeviceBuffer<ItemAoS> dItems;
    DeviceBuffer<int> dOut(n);
    dItems.upload(items.data(), n);
    KernelTimer timer;
    timer.start();
    if (n > 0) foodWaitAoS<<<blocksFor(n, kThreads), kThreads>>>(dItems.data(), nowMs, static_cast<int>(n), dOut.data());
    timing.kernelMs = timer.stop();
    out.assign(n, 0);
    dOut.download(out.data(), n);
    timing.e2eMs = msSince(t0);
    return timing;
}
}
