#include "Stencil.h"
#include <chrono>
#include <cstddef>
#include <utility>
#include "Cuda.h"

namespace stencil {
namespace {
constexpr int kBlock = 16;

__global__ void sweep(const float* sources, const float* u, float* next, int w, int h) {
    const int x = blockIdx.x * kBlock + threadIdx.x;
    const int y = blockIdx.y * kBlock + threadIdx.y;
    if (x >= w || y >= h) return;
    const int i = y * w + x;
    if (sources[i] > 0.0f) {
        next[i] = 1.0f;
        return;
    }
    const float up = y > 0 ? u[i - w] : 0.0f;
    const float down = y + 1 < h ? u[i + w] : 0.0f;
    const float left = x > 0 ? u[i - 1] : 0.0f;
    const float right = x + 1 < w ? u[i + 1] : 0.0f;
    next[i] = kDecay * (0.25f * (((up + down) + left) + right));
}

__global__ void seed(const float* sources, float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) u[i] = sources[i] > 0.0f ? 1.0f : 0.0f;
}
}

GpuTiming gpu(const Grid& sources, int sweeps, Grid& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const std::size_t n = sources.v.size();
    DeviceBuffer<float> dSources, a(n), b(n);
    dSources.upload(sources.v.data(), n);
    const dim3 block(kBlock, kBlock);
    const dim3 grid(blocksFor(static_cast<std::size_t>(sources.w), kBlock),
                    blocksFor(static_cast<std::size_t>(sources.h), kBlock));
    KernelTimer timer;
    timer.start();
    seed<<<blocksFor(n, 256), 256>>>(dSources.data(), a.data(), static_cast<int>(n));
    float* u = a.data();
    float* next = b.data();
    for (int s = 0; s < sweeps; ++s) {
        sweep<<<grid, block>>>(dSources.data(), u, next, sources.w, sources.h);
        std::swap(u, next);
    }
    timing.kernelMs = timer.stop();
    out = Grid(sources.w, sources.h);
    CUDA_CHECK(cudaMemcpy(out.v.data(), u, n * sizeof(float), cudaMemcpyDeviceToHost));
    timing.e2eMs = msSince(t0);
    return timing;
}
}
