#include "Argmin.h"
#include <chrono>
#include <cstddef>
#include "Cuda.h"
#include "EtaKernels.h"

namespace argmin {
namespace {
constexpr unsigned kThreads = 256;

__device__ __forceinline__ Key packOnDevice(int eta, int id) {
    return (static_cast<Key>(static_cast<unsigned>(eta)) << 32) | static_cast<unsigned>(id);
}

__global__ void rowMin(const int* eta, int cols, const int* ids, Key* out) {
    __shared__ Key partial[kThreads];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const long long base = static_cast<long long>(row) * cols;
    Key best = kNone;
    for (int j = t; j < cols; j += blockDim.x) {
        const Key k = packOnDevice(eta[base + j], ids[j]);
        if (k < best) best = k;
    }
    partial[t] = best;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride >= 32; stride /= 2) {
        if (t < stride && partial[t + stride] < partial[t]) partial[t] = partial[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        Key v = partial[t];
        for (int offset = 16; offset > 0; offset /= 2) {
            const Key other = __shfl_down_sync(0xffffffffu, v, offset);
            if (other < v) v = other;
        }
        if (t == 0) out[row] = v;
    }
}
}

GpuTiming gpu(const std::vector<int>& eta, int rows, int cols, const std::vector<int>& ids, std::vector<Key>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    DeviceBuffer<int> dEta, dIds;
    DeviceBuffer<Key> dOut(static_cast<std::size_t>(rows));
    dEta.upload(eta.data(), eta.size());
    dIds.upload(ids.data(), ids.size());
    KernelTimer timer;
    timer.start();
    rowMin<<<static_cast<unsigned>(rows), kThreads>>>(dEta.data(), cols, dIds.data(), dOut.data());
    timing.kernelMs = timer.stop();
    out.resize(static_cast<std::size_t>(rows));
    dOut.download(out.data(), out.size());
    timing.e2eMs = msSince(t0);
    return timing;
}

Key bestCourierGpu(const eta::DeviceTable& table, const eta::Params& p, const eta::Query& q,
                   const eta::Couriers& couriers, const std::vector<int>& ids) {
    const int k = static_cast<int>(couriers.size());
    if (k == 0) return kNone;
    DeviceBuffer<eta::Query> dQuery;
    DeviceBuffer<int> dNode, dIds, dEta(couriers.size());
    DeviceBuffer<long long> dFree;
    DeviceBuffer<Key> dOut(1);
    dQuery.upload(&q, 1);
    dNode.upload(couriers.freeNode.data(), couriers.size());
    dFree.upload(couriers.freeAtMs.data(), couriers.size());
    dIds.upload(ids.data(), ids.size());
    eta::launchSoA(table, p, dQuery.data(), 1, dNode.data(), dFree.data(), k, dEta.data());
    rowMin<<<1, kThreads>>>(dEta.data(), k, dIds.data(), dOut.data());
    CUDA_CHECK(cudaGetLastError());
    Key best = kNone;
    dOut.download(&best, 1);
    return best;
}
}
