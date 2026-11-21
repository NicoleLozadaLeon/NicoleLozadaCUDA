#include "FleetSummary.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include "Cuda.h"

namespace fleetsum {
namespace {
constexpr unsigned kThreads = 256;  // 8 warps
constexpr unsigned kWarp = 32;
constexpr unsigned kWarps = kThreads / kWarp;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr unsigned kCoarse = 4;       // couriers per thread before the tree (thread coarsening)
constexpr unsigned kMaxBlocks = 1024;

__device__ __forceinline__ Summary deviceIdentity() {
    Summary s{};
    s.mostItemsIdx = kNoIndex;
    s.latestEndIdx = kNoIndex;
    return s;
}

__device__ __forceinline__ bool beats(unsigned long long va, unsigned ia, unsigned long long vb, unsigned ib) {
    return va > vb || (va == vb && ia < ib);
}

__device__ __forceinline__ Summary combine(const Summary& a, const Summary& b) {
    Summary r = a;
    for (int k = 0; k < 4; ++k) r.state[k] += b.state[k];
    r.available += b.available;
    r.items += b.items;
    r.picked += b.picked;
    if (beats(b.mostItems, b.mostItemsIdx, a.mostItems, a.mostItemsIdx)) {
        r.mostItems = b.mostItems;
        r.mostItemsIdx = b.mostItemsIdx;
    }
    if (beats(b.latestEnd, b.latestEndIdx, a.latestEnd, a.latestEndIdx)) {
        r.latestEnd = b.latestEnd;
        r.latestEndIdx = b.latestEndIdx;
    }
    return r;
}

// Each field is shuffled on its own, as in bikeshare_kpis.cu.
__device__ __forceinline__ Summary shuffleDown(const Summary& v, unsigned delta) {
    Summary r;
    for (int k = 0; k < 4; ++k) r.state[k] = __shfl_down_sync(kFullMask, v.state[k], delta);
    r.available = __shfl_down_sync(kFullMask, v.available, delta);
    r.items = __shfl_down_sync(kFullMask, v.items, delta);
    r.picked = __shfl_down_sync(kFullMask, v.picked, delta);
    r.mostItems = __shfl_down_sync(kFullMask, v.mostItems, delta);
    r.mostItemsIdx = __shfl_down_sync(kFullMask, v.mostItemsIdx, delta);
    r.latestEnd = __shfl_down_sync(kFullMask, v.latestEnd, delta);
    r.latestEndIdx = __shfl_down_sync(kFullMask, v.latestEndIdx, delta);
    return r;
}

__device__ __forceinline__ Summary warpReduce(Summary v) {
    for (unsigned stride = kWarp / 2; stride > 0; stride /= 2) v = combine(v, shuffleDown(v, stride));
    return v;
}

// Block reduction, two stages (bikeshare_kpis.cu): every warp reduces its threads, lane 0 stores the warp
// summary, one barrier, and warp 0 reduces the warp summaries.
__device__ __forceinline__ Summary blockReduce(Summary v) {
    __shared__ Summary warp_s[kWarps];
    const unsigned warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    v = warpReduce(v);
    if (lane == 0) warp_s[warp] = v;
    __syncthreads();
    Summary total = deviceIdentity();
    if (warp == 0) {
        total = warpReduce(lane < kWarps ? warp_s[lane] : deviceIdentity());
    }
    return total;
}

// Final design: no branches. Each state adds a 0 or a 1; the argmax are selects.
__device__ __forceinline__ void addBranchFree(Summary& s, unsigned i, unsigned char state, unsigned char available,
                                              unsigned char items, unsigned char picked, long long legEnd) {
    s.state[0] += state == 0;
    s.state[1] += state == 1;
    s.state[2] += state == 2;
    s.state[3] += state == 3;
    s.available += available;
    s.items += items;
    s.picked += picked;
    const bool moreItems = beats(items, i, s.mostItems, s.mostItemsIdx);
    s.mostItems = moreItems ? items : s.mostItems;
    s.mostItemsIdx = moreItems ? i : s.mostItemsIdx;
    const auto end = static_cast<unsigned long long>(legEnd);
    const bool later = beats(end, i, s.latestEnd, s.latestEndIdx);
    s.latestEnd = later ? end : s.latestEnd;
    s.latestEndIdx = later ? i : s.latestEndIdx;
}

// For comparison: a switch on the state and ifs, one path per case.
__device__ __forceinline__ void addBranchy(Summary& s, unsigned i, unsigned char state, unsigned char available,
                                           unsigned char items, unsigned char picked, long long legEnd) {
    switch (state) {
        case 0: ++s.state[0]; break;
        case 1: ++s.state[1]; break;
        case 2: ++s.state[2]; break;
        default: ++s.state[3]; break;
    }
    if (available) ++s.available;
    if (items != 0) {
        s.items += items;
        s.picked += picked;
    }
    if (beats(items, i, s.mostItems, s.mostItemsIdx)) {
        s.mostItems = items;
        s.mostItemsIdx = i;
    }
    const auto end = static_cast<unsigned long long>(legEnd);
    if (beats(end, i, s.latestEnd, s.latestEndIdx)) {
        s.latestEnd = end;
        s.latestEndIdx = i;
    }
}

// One thread folds several couriers (a grid-stride loop), then the block tree. One partial summary per
// block and no atomics: a second kernel combines the partials in a fixed order.
template <bool BranchFree>
__device__ __forceinline__ void pass(const unsigned char* bytes, const long long* legEnd, unsigned n, Summary* partial) {
    const unsigned char *state = bytes, *available = bytes + n, *items = bytes + 2 * n, *picked = bytes + 3 * n;
    Summary s = deviceIdentity();
    for (unsigned i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) {
        if (BranchFree) addBranchFree(s, i, state[i], available[i], items[i], picked[i], legEnd[i]);
        else addBranchy(s, i, state[i], available[i], items[i], picked[i], legEnd[i]);
    }
    s = blockReduce(s);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

__global__ void fleetBranchFree(const unsigned char* bytes, const long long* legEnd, unsigned n, Summary* partial) {
    pass<true>(bytes, legEnd, n, partial);
}

__global__ void fleetBranchy(const unsigned char* bytes, const long long* legEnd, unsigned n, Summary* partial) {
    pass<false>(bytes, legEnd, n, partial);
}

// Second pass: one block combines the block summaries in a fixed order.
__global__ void fleetCombine(const Summary* partial, unsigned blocks, Summary* out) {
    Summary s = deviceIdentity();
    for (unsigned b = threadIdx.x; b < blocks; b += kThreads) s = combine(s, partial[b]);
    s = blockReduce(s);
    if (threadIdx.x == 0) *out = s;
}

template <bool BranchFree>
GpuTiming run(const FleetData& fleet, Summary& out) {
    const auto t0 = std::chrono::steady_clock::now();
    GpuTiming timing;
    const std::size_t n = fleet.size();
    if (n == 0) {
        out = identity();
        return timing;
    }
    // The four small arrays travel in one buffer: state | available | items | picked.
    std::vector<unsigned char> bytes;
    bytes.reserve(4 * n);
    for (const std::vector<unsigned char>* v : {&fleet.state, &fleet.available, &fleet.items, &fleet.picked})
        bytes.insert(bytes.end(), v->begin(), v->end());
    const unsigned blocks = std::max(1u, std::min(blocksFor(n, kThreads * kCoarse), kMaxBlocks));
    DeviceBuffer<unsigned char> dBytes;
    DeviceBuffer<long long> dEnd;
    DeviceBuffer<Summary> dPartial(blocks), dOut(1);
    dBytes.upload(bytes.data(), bytes.size());
    dEnd.upload(fleet.legEndMs.data(), n);
    KernelTimer timer;
    timer.start();
    if (BranchFree) fleetBranchFree<<<blocks, kThreads>>>(dBytes.data(), dEnd.data(), static_cast<unsigned>(n), dPartial.data());
    else fleetBranchy<<<blocks, kThreads>>>(dBytes.data(), dEnd.data(), static_cast<unsigned>(n), dPartial.data());
    fleetCombine<<<1, kThreads>>>(dPartial.data(), blocks, dOut.data());
    timing.kernelMs = timer.stop();
    dOut.download(&out, 1);
    timing.e2eMs = msSince(t0);
    return timing;
}
}

GpuTiming gpuBranchFree(const FleetData& fleet, Summary& out) { return run<true>(fleet, out); }
GpuTiming gpuBranchy(const FleetData& fleet, Summary& out) { return run<false>(fleet, out); }
}
