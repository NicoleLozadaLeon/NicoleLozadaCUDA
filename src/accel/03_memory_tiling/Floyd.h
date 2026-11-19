#pragma once
#include <vector>
#include "Gpu.h"

namespace floyd {
constexpr int kInf = 0x3fffffff;

struct Network {
    int n = 0;
    std::vector<int> rowStart;
    std::vector<int> to;
    std::vector<int> weight;
};

Network gridNetwork(int side, unsigned seed);
std::vector<int> denseWeights(const Network& net, int paddedN);
std::vector<int> dijkstraOne(const Network& net);
std::vector<int> dijkstraMulti(const Network& net, int threads);
GpuTiming floydGpu(const Network& net, std::vector<int>& dist);
}