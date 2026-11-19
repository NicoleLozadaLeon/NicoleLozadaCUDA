#include "Floyd.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include "Parallel.h"

namespace floyd {
namespace {
void dijkstraFrom(const Network& net, int source, int* row) {
    using Item = std::pair<int, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    for (int i = 0; i < net.n; ++i) row[i] = kInf;
    row[source] = 0;
    pq.push({0, source});
    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d > row[u]) continue;
        for (int e = net.rowStart[static_cast<std::size_t>(u)]; e < net.rowStart[static_cast<std::size_t>(u) + 1]; ++e) {
            const int v = net.to[static_cast<std::size_t>(e)];
            const int candidate = d + net.weight[static_cast<std::size_t>(e)];
            if (candidate < row[v]) {
                row[v] = candidate;
                pq.push({candidate, v});
            }
        }
    }
}
}

Network gridNetwork(int side, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> length(250, 400);
    const int n = side * side;
    std::vector<std::vector<std::pair<int, int>>> adj(static_cast<std::size_t>(n));
    auto id = [side](int r, int c) { return r * side + c; };
    for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c) {
            if (c + 1 < side) {
                const int w = length(rng);
                const bool oneWay = r % 3 == 1;
                const bool eastward = (r / 3) % 2 == 0;
                if (!oneWay || eastward) adj[static_cast<std::size_t>(id(r, c))].push_back({id(r, c + 1), w});
                if (!oneWay || !eastward) adj[static_cast<std::size_t>(id(r, c + 1))].push_back({id(r, c), w});
            }
            if (r + 1 < side) {
                const int w = length(rng);
                adj[static_cast<std::size_t>(id(r, c))].push_back({id(r + 1, c), w});
                adj[static_cast<std::size_t>(id(r + 1, c))].push_back({id(r, c), w});
            }
        }
    Network net;
    net.n = n;
    net.rowStart.push_back(0);
    for (const auto& edges : adj) {
        for (const auto& [to, w] : edges) {
            net.to.push_back(to);
            net.weight.push_back(w);
        }
        net.rowStart.push_back(static_cast<int>(net.to.size()));
    }
    return net;
}

std::vector<int> denseWeights(const Network& net, int paddedN) {
    const std::size_t N = static_cast<std::size_t>(paddedN);
    std::vector<int> w(N * N, kInf);
    for (std::size_t i = 0; i < N; ++i) w[i * N + i] = 0;
    for (int u = 0; u < net.n; ++u)
        for (int e = net.rowStart[static_cast<std::size_t>(u)]; e < net.rowStart[static_cast<std::size_t>(u) + 1]; ++e) {
            int& cell = w[static_cast<std::size_t>(u) * N + static_cast<std::size_t>(net.to[static_cast<std::size_t>(e)])];
            cell = std::min(cell, net.weight[static_cast<std::size_t>(e)]);
        }
    return w;
}

std::vector<int> dijkstraOne(const Network& net) {
    const std::size_t n = static_cast<std::size_t>(net.n);
    std::vector<int> dist(n * n);
    for (int s = 0; s < net.n; ++s) dijkstraFrom(net, s, dist.data() + static_cast<std::size_t>(s) * n);
    return dist;
}

std::vector<int> dijkstraMulti(const Network& net, int threads) {
    const std::size_t n = static_cast<std::size_t>(net.n);
    std::vector<int> dist(n * n);
    parallelFor(n, threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t s = begin; s < end; ++s) dijkstraFrom(net, static_cast<int>(s), dist.data() + s * n);
    });
    return dist;
}

#ifndef EQ_USE_CUDA
GpuTiming floydGpu(const Network&, std::vector<int>&) { throw std::runtime_error("built without CUDA"); }
#endif
}