#include "Nearby.h"
#include <chrono>
#include <cstddef>
#include <vector>
#include "Cuda.h"

namespace nearby {
namespace {
// 128 threads per block: with the 3600 corners of the city that is 29 blocks, about one per SM.
constexpr int kThreads = 128;

// One thread per corner. The block walks the couriers one tile of 128 at a time: every thread loads
// one courier into shared memory, the block waits, and then every thread compares its corner with
// the 128 couriers of the tile. Each courier is read from global memory once per block, not once
// per thread. Threads past the last corner still load and reach both barriers (as in
// 04_tiled_matmul_boundary): they just do not store.
__global__ void countTiled(const int* cx, const int* cy, int corners, const int* qx, const int* qy, int couriers,
                           int r2, int* out) {
    __shared__ int tileX[kThreads], tileY[kThreads];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = i < corners ? cx[i] : 0, y = i < corners ? cy[i] : 0;
    int count = 0;
    for (int base = 0; base < couriers; base += kThreads) {
        const int j = base + threadIdx.x;
        if (j < couriers) {
            tileX[threadIdx.x] = qx[j];
            tileY[threadIdx.x] = qy[j];
        }
        __syncthreads();
        const int inTile = min(kThreads, couriers - base);
        for (int k = 0; k < inTile; ++k) {
            const int dx = tileX[k] - x, dy = tileY[k] - y;
            count += dx * dx + dy * dy <= r2 ? 1 : 0;
        }
        __syncthreads();
    }
    if (i < corners) out[i] = count;
}

// The same count without tiles: every thread reads every courier from global memory.
__global__ void countNaive(const int* cx, const int* cy, int corners, const int* qx, const int* qy, int couriers,
                           int r2, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= corners) return;
    const int x = cx[i], y = cy[i];
    int count = 0;
    for (int j = 0; j < couriers; ++j) {
        const int dx = qx[j] - x, dy = qy[j] - y;
        count += dx * dx + dy * dy <= r2 ? 1 : 0;
    }
    out[i] = count;
}

template <bool Tiled>
void launch(const int* cx, const int* cy, int n, const int* qx, const int* qy, int m, int r2, int* out) {
    if (n <= 0) return;
    const unsigned blocks = blocksFor(static_cast<std::size_t>(n), kThreads);
    if (Tiled)
        countTiled<<<blocks, kThreads>>>(cx, cy, n, qx, qy, m, r2, out);
    else
        countNaive<<<blocks, kThreads>>>(cx, cy, n, qx, qy, m, r2, out);
}

// [x of every point | y of every point], so the points travel in one copy.
std::vector<int> packed(const std::vector<const std::vector<int>*>& parts) {
    std::vector<int> all;
    for (const std::vector<int>* v : parts) all.insert(all.end(), v->begin(), v->end());
    return all;
}

template <bool Tiled>
GpuTiming resident(const DeviceCorners& corners, const Points& couriers, int radiusM, std::vector<int>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const int n = corners.n(), m = static_cast<int>(couriers.x.size());
    const std::vector<int> q = packed({&couriers.x, &couriers.y});
    DeviceBuffer<int> dq;
    dq.upload(q.data(), q.size());
    KernelTimer timer;
    timer.start();
    launch<Tiled>(corners.x(), corners.y(), n, dq.data(), dq.data() + m, m, radiusM * radiusM, corners.out());
    timing.kernelMs = timer.stop();
    out.assign(static_cast<std::size_t>(n), 0);
    if (n > 0) CUDA_CHECK(cudaMemcpy(out.data(), corners.out(), out.size() * sizeof(int), cudaMemcpyDeviceToHost));
    timing.e2eMs = msSince(t0);
    return timing;
}
}

DeviceCorners::DeviceCorners(const Points& corners) : n_(static_cast<int>(corners.x.size())) {
    if (n_ == 0) return;
    const std::vector<int> c = packed({&corners.x, &corners.y});
    CUDA_CHECK(cudaMalloc(&data_, c.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(data_, c.data(), c.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&out_, static_cast<std::size_t>(n_) * sizeof(int)));
}

DeviceCorners::~DeviceCorners() {
    if (data_) cudaFree(data_);
    if (out_) cudaFree(out_);
}

GpuTiming gpuTiled(const DeviceCorners& corners, const Points& couriers, int radiusM, std::vector<int>& out) {
    return resident<true>(corners, couriers, radiusM, out);
}
GpuTiming gpuNaive(const DeviceCorners& corners, const Points& couriers, int radiusM, std::vector<int>& out) {
    return resident<false>(corners, couriers, radiusM, out);
}

GpuTiming gpuUploadAll(const Points& corners, const Points& couriers, int radiusM, std::vector<int>& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const int n = static_cast<int>(corners.x.size()), m = static_cast<int>(couriers.x.size());
    const std::vector<int> all = packed({&corners.x, &corners.y, &couriers.x, &couriers.y});
    DeviceBuffer<int> in, dOut(static_cast<std::size_t>(n));
    in.upload(all.data(), all.size());
    const int* cx = in.data();
    KernelTimer timer;
    timer.start();
    launch<true>(cx, cx + n, n, cx + 2 * n, cx + 2 * n + m, m, radiusM * radiusM, dOut.data());
    timing.kernelMs = timer.stop();
    out.assign(static_cast<std::size_t>(n), 0);
    dOut.download(out.data(), out.size());
    timing.e2eMs = msSince(t0);
    return timing;
}
}
