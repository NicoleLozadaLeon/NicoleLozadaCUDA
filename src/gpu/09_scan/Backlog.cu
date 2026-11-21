#include "Backlog.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include "Cuda.h"

// The kernels follow ec2/09_scan/bikeshare_rentals_in_progress.cu: the coarsened three-phase scan of one section
// (a coalesced load, a scan of each thread's run, Kogge-Stone over the run totals, and the totals added to the
// left), the hierarchical scan for any length, an atomicMax reduction for the peak and an atomicMin search for the
// first event that reached it.
namespace backlog {
namespace {
constexpr int kThreads = 256;
constexpr int kCoarsen = 8;
constexpr int kSection = kThreads * kCoarsen;  // 2048 events per block
constexpr int kOffsetThreads = 256;
constexpr int kReduceThreads = 256;
constexpr int kMaxReduceBlocks = 1024;

__device__ void loadSection(const int* X, int n, int base, int* XY) {
    for (int k = 0; k < kCoarsen; ++k) {
        const int j = k * kThreads + threadIdx.x;
        XY[j] = (base + j < n) ? X[base + j] : 0;
    }
}

__device__ void storeSection(const int* XY, int n, int base, int* Y) {
    for (int k = 0; k < kCoarsen; ++k) {
        const int j = k * kThreads + threadIdx.x;
        if (base + j < n) Y[base + j] = XY[j];
    }
}

// Phase 1: every thread scans its own run of kCoarsen consecutive elements.
__device__ void scanOwnRun(int* XY, int first) {
    for (int k = 1; k < kCoarsen; ++k) XY[first + k] += XY[first + k - 1];
}

// Phase 2: Kogge-Stone over the totals of the runs.
__device__ void koggeStoneOverTotals(int* part) {
    const int t = threadIdx.x;
    for (int stride = 1; stride < kThreads; stride *= 2) {
        __syncthreads();
        int temp = part[t];
        if (t >= stride) temp += part[t - stride];
        __syncthreads();
        part[t] = temp;
    }
}

// Phase 3: every thread adds the total of the runs to its left.
__device__ void addTotalOfRunsToTheLeft(int* XY, const int* part, int first) {
    if (threadIdx.x > 0) {
        const int add = part[threadIdx.x - 1];
        for (int k = 0; k < kCoarsen; ++k) XY[first + k] += add;
    }
}

__global__ void backlogScanSections(const int* X, int* Y, int* sums, int n) {
    __shared__ int XY[kSection];
    __shared__ int part[kThreads];
    const int t = threadIdx.x;
    const int base = blockIdx.x * kSection;
    const int first = t * kCoarsen;

    loadSection(X, n, base, XY);
    __syncthreads();

    scanOwnRun(XY, first);
    part[t] = XY[first + kCoarsen - 1];

    koggeStoneOverTotals(part);
    __syncthreads();

    addTotalOfRunsToTheLeft(XY, part, first);
    __syncthreads();

    storeSection(XY, n, base, Y);
    if (t == 0) sums[blockIdx.x] = part[kThreads - 1];
}

__global__ void backlogAddOffsets(int* Y, const int* scannedSums, int n, int section) {
    if (blockIdx.x == 0) return;
    const int add = scannedSums[blockIdx.x - 1];
    for (int j = threadIdx.x; j < section; j += blockDim.x) {
        const long long idx = static_cast<long long>(blockIdx.x) * section + j;
        if (idx < n) Y[idx] += add;
    }
}

__global__ void backlogInitPeak(int* peak, int* first) {
    *peak = INT_MIN;
    *first = INT_MAX;
}

__global__ void backlogMaxValue(const int* v, int* result, int n) {
    __shared__ int blockMax;
    if (threadIdx.x == 0) blockMax = INT_MIN;
    __syncthreads();

    int threadMax = INT_MIN;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) threadMax = max(threadMax, v[i]);
    atomicMax(&blockMax, threadMax);
    __syncthreads();

    if (threadIdx.x == 0) atomicMax(result, blockMax);
}

// The peak value stays on the device: the kernel reads it, so the host does not wait in between.
__global__ void backlogFirstIndex(const int* v, const int* value, int* result, int n) {
    const int stride = blockDim.x * gridDim.x;
    const int target = *value;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        if (v[i] == target) atomicMin(result, i);
}

int scratchInts(int n) {
    const int blocks = (n + kSection - 1) / kSection;
    return blocks == 1 ? 1 : 2 * blocks + scratchInts(blocks);
}

void scanInclusive(const int* in, int* out, int n, int* scratch) {
    const int blocks = (n + kSection - 1) / kSection;
    int* blockSums = scratch;
    backlogScanSections<<<blocks, kThreads>>>(in, out, blockSums, n);
    if (blocks > 1) {
        int* scannedSums = scratch + blocks;
        scanInclusive(blockSums, scannedSums, blocks, scratch + 2 * blocks);
        backlogAddOffsets<<<blocks, kOffsetThreads>>>(out, scannedSums, n, kSection);
    }
}
}

GpuTiming gpuScan(const EventStream& events, Result& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    out = Result{};
    if (events.size() == 0) return timing;
    const int n = static_cast<int>(events.size());
    DeviceBuffer<int> dDeltas, dWaiting(events.size()), dScratch(static_cast<std::size_t>(scratchInts(n))), dPeak(2);
    dDeltas.upload(events.deltas.data(), events.size());
    const int reduceBlocks = std::min((n + kReduceThreads - 1) / kReduceThreads, kMaxReduceBlocks);
    KernelTimer timer;
    timer.start();
    backlogInitPeak<<<1, 1>>>(dPeak.data(), dPeak.data() + 1);
    scanInclusive(dDeltas.data(), dWaiting.data(), n, dScratch.data());
    backlogMaxValue<<<reduceBlocks, kReduceThreads>>>(dWaiting.data(), dPeak.data(), n);
    backlogFirstIndex<<<reduceBlocks, kReduceThreads>>>(dWaiting.data(), dPeak.data(), dPeak.data() + 1, n);
    timing.kernelMs = timer.stop();
    out.waiting.resize(events.size());
    dWaiting.download(out.waiting.data(), events.size());
    int ends[2] = {0, -1};
    dPeak.download(ends, 2);
    out.peak = ends[0];
    out.firstPeak = ends[1];
    timing.e2eMs = msSince(t0);
    return timing;
}
}
