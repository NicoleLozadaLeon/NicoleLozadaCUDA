#include "Hotspots.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include "Cuda.h"

// The smoothing follows ec2/05_conv/bikeshare_demand_heatmap.cu: the 13 x 13 weights in constant memory, one
// 32 x 32 shared-memory tile per block, and only the inner 20 x 20 threads write (the other 12 on each side only
// load the halo). The orders are counted with atomicAdd, as in the histogram kernels of ec2/07_histogram.
namespace hotspots {
namespace {
constexpr int kInTile = 32;
constexpr int kOutTile = kInTile - 2 * kRadius;  // 20
constexpr int kDirectBlock = 16;
constexpr int kCountThreads = 256;
__constant__ float cFilter[kSide * kSide];

__global__ void countOrders(const int* cells, int n, float* counts, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    if (c >= 0 && c < size) atomicAdd(&counts[c], 1.0f);
}

__global__ void __launch_bounds__(kInTile * kInTile) smoothTiled(const float* in, float* out, int w, int h) {
    __shared__ float tile[kInTile][kInTile];
    const int col = blockIdx.x * kOutTile + static_cast<int>(threadIdx.x) - kRadius;
    const int row = blockIdx.y * kOutTile + static_cast<int>(threadIdx.y) - kRadius;
    const bool inside = col >= 0 && col < w && row >= 0 && row < h;
    tile[threadIdx.y][threadIdx.x] = inside ? in[row * w + col] : 0.0f;
    __syncthreads();

    const int tx = static_cast<int>(threadIdx.x) - kRadius;
    const int ty = static_cast<int>(threadIdx.y) - kRadius;
    if (inside && tx >= 0 && tx < kOutTile && ty >= 0 && ty < kOutTile) {
        float sum = 0.0f;
#pragma unroll
        for (int fy = 0; fy < kSide; ++fy) {
#pragma unroll
            for (int fx = 0; fx < kSide; ++fx) sum += cFilter[fy * kSide + fx] * tile[ty + fy][tx + fx];
        }
        out[row * w + col] = sum;
    }
}

__global__ void smoothDirect(const float* in, float* out, int w, int h) {
    const int x = blockIdx.x * kDirectBlock + threadIdx.x;
    const int y = blockIdx.y * kDirectBlock + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
#pragma unroll
    for (int fy = 0; fy < kSide; ++fy) {
        const int yy = y - kRadius + fy;
        if (yy < 0 || yy >= h) continue;
#pragma unroll
        for (int fx = 0; fx < kSide; ++fx) {
            const int xx = x - kRadius + fx;
            if (xx < 0 || xx >= w) continue;
            sum += cFilter[fy * kSide + fx] * in[yy * w + xx];
        }
    }
    out[y * w + x] = sum;
}

GpuTiming run(bool tiled, int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, Grid& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const std::size_t size = static_cast<std::size_t>(w) * h;
    DeviceBuffer<int> dCells;
    DeviceBuffer<float> dCounts(size), dSmooth(size);
    if (!cells.empty()) dCells.upload(cells.data(), cells.size());
    CUDA_CHECK(cudaMemcpyToSymbol(cFilter, filter.data(), filter.size() * sizeof(float)));
    KernelTimer timer;
    timer.start();
    dCounts.zero();
    if (!cells.empty())
        countOrders<<<blocksFor(cells.size(), kCountThreads), kCountThreads>>>(
            dCells.data(), static_cast<int>(cells.size()), dCounts.data(), static_cast<int>(size));
    if (tiled) {
        const dim3 grid(blocksFor(static_cast<std::size_t>(w), kOutTile), blocksFor(static_cast<std::size_t>(h), kOutTile));
        smoothTiled<<<grid, dim3(kInTile, kInTile)>>>(dCounts.data(), dSmooth.data(), w, h);
    } else {
        const dim3 grid(blocksFor(static_cast<std::size_t>(w), kDirectBlock),
                        blocksFor(static_cast<std::size_t>(h), kDirectBlock));
        smoothDirect<<<grid, dim3(kDirectBlock, kDirectBlock)>>>(dCounts.data(), dSmooth.data(), w, h);
    }
    timing.kernelMs = timer.stop();
    out = Grid(w, h);
    dSmooth.download(out.v.data(), size);
    timing.e2eMs = msSince(t0);
    return timing;
}
}

namespace {
__global__ void spin(unsigned long long ns, float* sink) {
    unsigned long long start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    float x = static_cast<float>(threadIdx.x);
    do {
#pragma unroll
        for (int i = 0; i < 64; ++i) x = fmaf(x, 1.0001f, 0.5f);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < ns);
    if (x == -1.0f) *sink = x;  // never true: keeps the loop from being removed
}
}

void gpuWarmUp(int ms) {
    DeviceBuffer<float> sink(1);
    spin<<<std::max(gpu::smCount(), 1) * 4, 256>>>(static_cast<unsigned long long>(ms) * 1000000ull, sink.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

GpuTiming gpuTiled(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, Grid& out) {
    return run(true, w, h, cells, filter, out);
}

GpuTiming gpuDirect(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, Grid& out) {
    return run(false, w, h, cells, filter, out);
}
}
