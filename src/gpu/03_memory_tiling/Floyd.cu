#include "Floyd.h"
#include <chrono>
#include <cstddef>
#include "Cuda.h"

namespace floyd {
namespace {
constexpr int kTile = 32;

__device__ __forceinline__ int at(const int* d, int n, int row, int col) { return d[row * n + col]; }

__global__ void diagonalTile(int* d, int n, int k) {
    __shared__ int t[kTile][kTile];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int base = k * kTile;
    t[ty][tx] = at(d, n, base + ty, base + tx);
    __syncthreads();
    for (int m = 0; m < kTile; ++m) {
        const int through = t[ty][m] + t[m][tx];
        __syncthreads();
        if (through < t[ty][tx]) t[ty][tx] = through;
        __syncthreads();
    }
    d[(base + ty) * n + base + tx] = t[ty][tx];
}

__global__ void pivotRowAndColumn(int* d, int n, int k) {
    const int b = blockIdx.x;
    if (b == k) return;
    __shared__ int pivot[kTile][kTile];
    __shared__ int own[kTile][kTile];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int pk = k * kTile;
    const bool rowTile = blockIdx.y == 0;
    const int row = rowTile ? pk + ty : b * kTile + ty;
    const int col = rowTile ? b * kTile + tx : pk + tx;
    pivot[ty][tx] = at(d, n, pk + ty, pk + tx);
    own[ty][tx] = at(d, n, row, col);
    __syncthreads();
    for (int m = 0; m < kTile; ++m) {
        const int through = rowTile ? pivot[ty][m] + own[m][tx] : own[ty][m] + pivot[m][tx];
        __syncthreads();
        if (through < own[ty][tx]) own[ty][tx] = through;
        __syncthreads();
    }
    d[row * n + col] = own[ty][tx];
}

__global__ void remainingTiles(int* d, int n, int k) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (bi == k || bj == k) return;
    __shared__ int left[kTile][kTile];
    __shared__ int up[kTile][kTile];
    const int ty = threadIdx.y, tx = threadIdx.x;
    const int row = bi * kTile + ty, col = bj * kTile + tx;
    const int pk = k * kTile;
    left[ty][tx] = at(d, n, row, pk + tx);
    up[ty][tx] = at(d, n, pk + ty, col);
    __syncthreads();
    int best = at(d, n, row, col);
#pragma unroll 8
    for (int m = 0; m < kTile; ++m) best = min(best, left[ty][m] + up[m][tx]);
    d[row * n + col] = best;
}
}

GpuTiming floydGpu(const Network& net, std::vector<int>& dist) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const int blocks = (net.n + kTile - 1) / kTile;
    const int N = blocks * kTile;
    const std::vector<int> w = denseWeights(net, N);
    DeviceBuffer<int> d;
    d.upload(w.data(), w.size());
    KernelTimer timer;
    timer.start();
    const dim3 threads(kTile, kTile);
    for (int k = 0; k < blocks; ++k) {
        diagonalTile<<<1, threads>>>(d.data(), N, k);
        pivotRowAndColumn<<<dim3(blocks, 2), threads>>>(d.data(), N, k);
        remainingTiles<<<dim3(blocks, blocks), threads>>>(d.data(), N, k);
    }
    timing.kernelMs = timer.stop();
    std::vector<int> padded(w.size());
    d.download(padded.data(), padded.size());
    const std::size_t n = static_cast<std::size_t>(net.n);
    dist.assign(n * n, kInf);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const int v = padded[i * static_cast<std::size_t>(N) + j];
            dist[i * n + j] = v >= kInf ? kInf : v;
        }
    timing.e2eMs = msSince(t0);
    return timing;
}
}