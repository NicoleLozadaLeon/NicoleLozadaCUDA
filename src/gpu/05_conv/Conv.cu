#include "Conv.h"
#include <chrono>
#include <cstddef>
#include "Cuda.h"

namespace conv {
namespace {
constexpr int kBlock = 16;
__constant__ float cFilter[(2 * kMaxRadius + 1) * (2 * kMaxRadius + 1)];

template <int R>
__global__ void convolve(const float* in, float* out, int w, int h) {
    const int x = blockIdx.x * kBlock + threadIdx.x;
    const int y = blockIdx.y * kBlock + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
#pragma unroll
    for (int fy = -R; fy <= R; ++fy) {
        const int yy = y + fy;
        if (yy < 0 || yy >= h) continue;
#pragma unroll
        for (int fx = -R; fx <= R; ++fx) {
            const int xx = x + fx;
            if (xx < 0 || xx >= w) continue;
            sum += cFilter[(fy + R) * (2 * R + 1) + fx + R] * in[yy * w + xx];
        }
    }
    out[y * w + x] = sum;
}

void launch(int radius, dim3 grid, dim3 block, const float* in, float* out, int w, int h) {
    switch (radius) {
        case 1: convolve<1><<<grid, block>>>(in, out, w, h); break;
        case 2: convolve<2><<<grid, block>>>(in, out, w, h); break;
        case 3: convolve<3><<<grid, block>>>(in, out, w, h); break;
        case 4: convolve<4><<<grid, block>>>(in, out, w, h); break;
        case 5: convolve<5><<<grid, block>>>(in, out, w, h); break;
        default: convolve<6><<<grid, block>>>(in, out, w, h); break;
    }
}
}

GpuTiming gpu(const Grid& in, const std::vector<float>& filter, int radius, Grid& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    DeviceBuffer<float> dIn, dOut(in.v.size());
    dIn.upload(in.v.data(), in.v.size());
    CUDA_CHECK(cudaMemcpyToSymbol(cFilter, filter.data(), filter.size() * sizeof(float)));
    const dim3 block(kBlock, kBlock);
    const dim3 grid(blocksFor(static_cast<std::size_t>(in.w), kBlock), blocksFor(static_cast<std::size_t>(in.h), kBlock));
    KernelTimer timer;
    timer.start();
    launch(radius, grid, block, dIn.data(), dOut.data(), in.w, in.h);
    timing.kernelMs = timer.stop();
    out = Grid(in.w, in.h);
    dOut.download(out.v.data(), out.v.size());
    timing.e2eMs = msSince(t0);
    return timing;
}
}
