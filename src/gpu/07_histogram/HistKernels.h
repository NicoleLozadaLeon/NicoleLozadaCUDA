#pragma once
#include <cstddef>
#include <stdexcept>
#include "Cuda.h"
#include "Gpu.h"
#include "Histogram.h"

namespace hist {
constexpr unsigned kThreads = 256;

template <class BinOf>
__global__ void privatized(long long n, int bins, BinOf binOf, unsigned* out) {
    extern __shared__ unsigned local[];
    for (int b = threadIdx.x; b < bins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&local[binOf(i)], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < bins; b += blockDim.x)
        if (local[b] != 0) atomicAdd(&out[b], local[b]);
}

template <class BinOf>
void launchPrivatized(long long n, int bins, BinOf binOf, unsigned* out) {
    if (bins > kMaxSharedBins) throw std::runtime_error("too many bins for the shared-memory copy");
    const unsigned wanted = blocksFor(static_cast<std::size_t>(n), kThreads);
    const unsigned cap = static_cast<unsigned>(gpu::smCount() * 8);
    const unsigned blocks = wanted == 0 ? 1 : (wanted < cap ? wanted : cap);
    privatized<<<blocks, kThreads, static_cast<std::size_t>(bins) * sizeof(unsigned)>>>(n, bins, binOf, out);
}

void launchByWidth(const int* values, long long n, int width, int bins, unsigned* out);
}
