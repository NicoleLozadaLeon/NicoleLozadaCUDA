#include "Graph.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <queue>
#include <utility>

namespace {

// Great-circle distance in meters between two lat/lon points.
double haversine_m(double lat1, double lon1, double lat2, double lon2) {
    const double kEarthRadius = 6371000.0;
    const double kDegToRad = 3.14159265358979323846 / 180.0;
    const double p1 = lat1 * kDegToRad;
    const double p2 = lat2 * kDegToRad;
    const double dp = p2 - p1;
    const double dl = (lon2 - lon1) * kDegToRad;
    const double a = std::sin(dp / 2) * std::sin(dp / 2) +
                     std::cos(p1) * std::cos(p2) * std::sin(dl / 2) * std::sin(dl / 2);
    return 2 * kEarthRadius * std::asin(std::sqrt(a));
}

}  // namespace

Graph::Graph(const Config& cfg) {
    const int n = static_cast<int>(cfg.nodes.size());
    ids_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const NodeDef& nd = cfg.nodes[static_cast<std::size_t>(i)];
        ids_.push_back(nd.id);
        index_[nd.id] = i;
        lat_.push_back(nd.lat);
        lon_.push_back(nd.lon);
    }

    adj_.assign(static_cast<std::size_t>(n), std::vector<Edge>());
    for (std::size_t s = 0; s < cfg.streets.size(); ++s) {
        const StreetDef& st = cfg.streets[s];
        const int a = index_.at(st.from);
        const int b = index_.at(st.to);
        const double meters = haversine_m(lat_[static_cast<std::size_t>(a)], lon_[static_cast<std::size_t>(a)],
                                          lat_[static_cast<std::size_t>(b)], lon_[static_cast<std::size_t>(b)]);
        adj_[static_cast<std::size_t>(a)].push_back({b, meters, static_cast<int>(s)});
        if (!st.oneWay)
            adj_[static_cast<std::size_t>(b)].push_back({a, meters, static_cast<int>(s)});
    }
    const std::size_t N = static_cast<std::size_t>(n);
    dist_.assign(N * N, kUnreachable);
    next_.assign(N * N, -1);
    using Item = std::pair<double, int>;
    for (int s = 0; s < n; ++s) {
        const std::size_t row = static_cast<std::size_t>(s) * N;
        std::vector<int> first(N, -1);
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
        dist_[row + static_cast<std::size_t>(s)] = 0.0;
        pq.push({0.0, s});
        while (!pq.empty()) {
            const Item top = pq.top();
            pq.pop();
            const double d = top.first;
            const int u = top.second;
            if (d > dist_[row + static_cast<std::size_t>(u)]) continue;   // stale entry
            for (const Edge& e : adj_[static_cast<std::size_t>(u)]) {
                const double candidate = d + e.meters;
                if (candidate < dist_[row + static_cast<std::size_t>(e.to)]) {
                    dist_[row + static_cast<std::size_t>(e.to)] = candidate;
                    first[static_cast<std::size_t>(e.to)] = (u == s) ? e.to : first[static_cast<std::size_t>(u)];
                    next_[row + static_cast<std::size_t>(e.to)] = first[static_cast<std::size_t>(e.to)];
                    pq.push({candidate, e.to});
                }
            }
        }
    }
}

int Graph::nodeCount() const { return static_cast<int>(ids_.size()); }
int Graph::indexOf(const std::string& id) const { return index_.at(id); }
const std::string& Graph::idOf(int node) const { return ids_[static_cast<std::size_t>(node)]; }
double Graph::lat(int node) const { return lat_[static_cast<std::size_t>(node)]; }
double Graph::lon(int node) const { return lon_[static_cast<std::size_t>(node)]; }
const std::vector<Graph::Edge>& Graph::edgesFrom(int node) const {
    return adj_[static_cast<std::size_t>(node)];
}

double Graph::distance(int from, int to) const {
    return dist_[static_cast<std::size_t>(from) * ids_.size() + static_cast<std::size_t>(to)];
}

bool Graph::reachable(int from, int to) const { return distance(from, to) < kUnreachable; }

std::vector<int> Graph::route(int from, int to) const {
    std::vector<int> path;
    if (!reachable(from, to)) return path;
    int cur = from;
    path.push_back(cur);
    while (cur != to) {
        cur = next_[static_cast<std::size_t>(cur) * ids_.size() + static_cast<std::size_t>(to)];
        path.push_back(cur);
    }
    return path;
}
