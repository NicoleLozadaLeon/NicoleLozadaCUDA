#include "Eta.h"
#include <chrono>
#include <cstddef>
#include "Cuda.h"
#include "EtaKernels.h"
#include "Floyd.h"

namespace eta {
namespace {
constexpr unsigned kThreads = 256;

__device__ __forceinline__ int etaOnDevice(const int* dist, Params p, Query q, int freeNode, long long freeAtMs) {
    const long long n = p.n;
    const int toRestaurant = dist[freeNode * n + q.restaurantNode];
    const int toCustomer = dist[q.restaurantNode * n + q.deliveryNode];
    if (toRestaurant >= floyd::kInf || toCustomer >= floyd::kInf) return kNoEta;
    long long t = max(p.nowMs, freeAtMs) + static_cast<long long>(toRestaurant) * 360000LL / p.metersPerHour;
    t = max(t, q.readyAtMs) + p.handoverMs;
    t += static_cast<long long>(toCustomer) * 360000LL / p.metersPerHour;
    const long long eta = t - p.nowMs;
    return static_cast<int>(eta < 0 ? 0 : (eta > kNoEta - 1 ? kNoEta - 1 : eta));
}

__global__ void etaSoA(const int* dist, Params p, const Query* orders, int m, const int* freeNode,
                       const long long* freeAtMs, int k, int* out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j >= k || i >= m) return;
    out[static_cast<long long>(i) * k + j] = etaOnDevice(dist, p, orders[i], freeNode[j], freeAtMs[j]);
}

__global__ void etaAoS(const int* dist, Params p, const Query* orders, int m, const CourierAoS* couriers, int k,
                       int* out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j >= k || i >= m) return;
    const CourierAoS c = couriers[j];
    out[static_cast<long long>(i) * k + j] = etaOnDevice(dist, p, orders[i], c.freeNode, c.freeAtMs);
}
}

DeviceTable::DeviceTable(const std::vector<int>& dist, int n) : n_(n) {
    CUDA_CHECK(cudaMalloc(&data_, dist.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(data_, dist.data(), dist.size() * sizeof(int), cudaMemcpyHostToDevice));
}

DeviceTable::~DeviceTable() {
    if (data_) cudaFree(data_);
}

void launchSoA(const DeviceTable& table, const Params& p, const Query* orders, int m, const int* freeNode,
               const long long* freeAtMs, int k, int* out) {
    etaSoA<<<dim3(blocksFor(static_cast<std::size_t>(k), kThreads), static_cast<unsigned>(m)), kThreads>>>(
        table.data(), p, orders, m, freeNode, freeAtMs, k, out);
}

GpuTiming gpuSoA(const DeviceTable& table, const Params& p, const std::vector<Query>& orders,
                 const Couriers& couriers, std::vector<int>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const int m = static_cast<int>(orders.size()), k = static_cast<int>(couriers.size());
    DeviceBuffer<Query> dOrders;
    DeviceBuffer<int> dNode;
    DeviceBuffer<long long> dFree;
    DeviceBuffer<int> dOut(orders.size() * couriers.size());
    dOrders.upload(orders.data(), orders.size());
    dNode.upload(couriers.freeNode.data(), couriers.size());
    dFree.upload(couriers.freeAtMs.data(), couriers.size());
    KernelTimer timer;
    timer.start();
    launchSoA(table, p, dOrders.data(), m, dNode.data(), dFree.data(), k, dOut.data());
    timing.kernelMs = timer.stop();
    out.resize(dOut.size());
    dOut.download(out.data(), out.size());
    timing.e2eMs = msSince(t0);
    return timing;
}

GpuTiming gpuAoS(const DeviceTable& table, const Params& p, const std::vector<Query>& orders,
                 const std::vector<CourierAoS>& couriers, std::vector<int>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const int m = static_cast<int>(orders.size()), k = static_cast<int>(couriers.size());
    DeviceBuffer<Query> dOrders;
    DeviceBuffer<CourierAoS> dCouriers;
    DeviceBuffer<int> dOut(orders.size() * couriers.size());
    dOrders.upload(orders.data(), orders.size());
    dCouriers.upload(couriers.data(), couriers.size());
    KernelTimer timer;
    timer.start();
    etaAoS<<<dim3(blocksFor(static_cast<std::size_t>(k), kThreads), static_cast<unsigned>(m)), kThreads>>>(
        table.data(), p, dOrders.data(), m, dCouriers.data(), k, dOut.data());
    timing.kernelMs = timer.stop();
    out.resize(dOut.size());
    dOut.download(out.data(), out.size());
    timing.e2eMs = msSince(t0);
    return timing;
}
}
