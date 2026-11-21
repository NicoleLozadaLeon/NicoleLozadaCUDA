#include "Scan.h"
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include "Cuda.h"
#include "ScanKernels.h"

namespace scan {
namespace {
constexpr int kScanThreads = 512;
constexpr int kScanItems = 2 * kScanThreads;
constexpr int kChunk = 1024;
constexpr int kGroupThreads = kMaxKeys;

__global__ void scanBlocks(unsigned* data, long long n, unsigned* sums) {
    __shared__ unsigned s[kScanItems];
    const int t = threadIdx.x;
    const long long base = static_cast<long long>(blockIdx.x) * kScanItems;
    const long long a = base + t, b = base + t + kScanThreads;
    s[t] = a < n ? data[a] : 0u;
    s[t + kScanThreads] = b < n ? data[b] : 0u;
    int offset = 1;
    for (int d = kScanItems >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            const int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            s[bi] += s[ai];
        }
        offset <<= 1;
    }
    __syncthreads();
    if (t == 0) {
        if (sums) sums[blockIdx.x] = s[kScanItems - 1];
        s[kScanItems - 1] = 0;
    }
    for (int d = 1; d < kScanItems; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            const int ai = offset * (2 * t + 1) - 1, bi = offset * (2 * t + 2) - 1;
            const unsigned left = s[ai];
            s[ai] = s[bi];
            s[bi] += left;
        }
    }
    __syncthreads();
    if (a < n) data[a] = s[t];
    if (b < n) data[b] = s[t + kScanThreads];
}

__global__ void addBlockSums(unsigned* data, long long n, const unsigned* sums) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) data[i] += sums[i / kScanItems];
}

long long blocksOf(long long n) { return (n + kScanItems - 1) / kScanItems; }

__device__ void loadChunk(const int* keys, long long n, int* s) {
    const long long base = static_cast<long long>(blockIdx.x) * kChunk;
    for (int i = threadIdx.x; i < kChunk; i += blockDim.x) s[i] = base + i < n ? keys[base + i] : -1;
    __syncthreads();
}

__global__ void countChunks(const int* keys, long long n, int keyCount, int chunks, unsigned* counts) {
    __shared__ int s[kChunk];
    loadChunk(keys, n, s);
    const int r = threadIdx.x;
    if (r >= keyCount) return;
    unsigned c = 0;
    for (int i = 0; i < kChunk; ++i) c += s[i] == r ? 1u : 0u;
    counts[static_cast<long long>(r) * chunks + blockIdx.x] = c;
}

__global__ void scatterChunks(const int* keys, long long n, int keyCount, int chunks, const unsigned* starts,
                              int* order) {
    __shared__ int s[kChunk];
    loadChunk(keys, n, s);
    const int r = threadIdx.x;
    if (r >= keyCount) return;
    const long long base = static_cast<long long>(blockIdx.x) * kChunk;
    unsigned pos = starts[static_cast<long long>(r) * chunks + blockIdx.x];
    for (int i = 0; i < kChunk; ++i)
        if (s[i] == r) order[pos++] = static_cast<int>(base + i);
}

__global__ void keyOffsets(const unsigned* starts, int keyCount, int chunks, long long n, unsigned* offsets) {
    const int r = threadIdx.x;
    if (r < keyCount) offsets[r] = starts[static_cast<long long>(r) * chunks];
    if (r == keyCount) offsets[r] = static_cast<unsigned>(n);
}
}

long long scratchFor(long long n) {
    const long long blocks = blocksOf(n);
    return blocks <= 1 ? 0 : blocks + scratchFor(blocks);
}

void exclusiveScan(unsigned* data, long long n, unsigned* scratch) {
    const long long blocks = blocksOf(n);
    if (blocks <= 1) {
        scanBlocks<<<1, kScanThreads>>>(data, n, nullptr);
        return;
    }
    scanBlocks<<<static_cast<unsigned>(blocks), kScanThreads>>>(data, n, scratch);
    exclusiveScan(scratch, blocks, scratch + blocks);
    addBlockSums<<<blocksFor(static_cast<std::size_t>(n), 256), 256>>>(data, n, scratch);
}

GpuTiming groupGpu(const std::vector<int>& keys, int keyCount, Grouping& out) {
    if (keyCount > kMaxKeys) throw std::runtime_error("groupGpu supports at most 256 keys");
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const long long n = static_cast<long long>(keys.size());
    const int chunks = static_cast<int>((n + kChunk - 1) / kChunk);
    const long long cells = static_cast<long long>(keyCount) * chunks;
    DeviceBuffer<int> dKeys, dOrder(keys.size());
    DeviceBuffer<unsigned> dStarts(static_cast<std::size_t>(cells)),
        dScratch(static_cast<std::size_t>(scratchFor(cells))), dOffsets(static_cast<std::size_t>(keyCount) + 1);
    dKeys.upload(keys.data(), keys.size());
    KernelTimer timer;
    timer.start();
    if (chunks > 0) {
        countChunks<<<static_cast<unsigned>(chunks), kGroupThreads>>>(dKeys.data(), n, keyCount, chunks, dStarts.data());
        exclusiveScan(dStarts.data(), cells, dScratch.data());
        scatterChunks<<<static_cast<unsigned>(chunks), kGroupThreads>>>(dKeys.data(), n, keyCount, chunks,
                                                                        dStarts.data(), dOrder.data());
        keyOffsets<<<1, kGroupThreads + 32>>>(dStarts.data(), keyCount, chunks, n, dOffsets.data());
    } else {
        dOffsets.zero();
    }
    timing.kernelMs = timer.stop();
    out.offsets.resize(static_cast<std::size_t>(keyCount) + 1);
    out.order.resize(keys.size());
    dOffsets.download(out.offsets.data(), out.offsets.size());
    dOrder.download(out.order.data(), out.order.size());
    timing.e2eMs = msSince(t0);
    return timing;
}
}
