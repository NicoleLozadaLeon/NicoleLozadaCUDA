#pragma once
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "Config.h"

// The street network. Built once, before any thread starts; afterwards every
// method is const, so any number of threads may read it without a lock.
class Graph {
public:
    static constexpr double kUnreachable = std::numeric_limits<double>::infinity();

    struct Edge {
        int to;
        double meters;
        int street;   // index into Config::streets
    };

    explicit Graph(const Config& cfg);

    int nodeCount() const;
    int indexOf(const std::string& id) const;   // throws std::out_of_range if unknown
    const std::string& idOf(int node) const;
    double lat(int node) const;
    double lon(int node) const;
    const std::vector<Edge>& edgesFrom(int node) const;

    // Meters along the streets, following one-way rules; kUnreachable if no path.
    double distance(int from, int to) const;
    bool reachable(int from, int to) const;
    // Nodes from `from` to `to`, both included; empty if unreachable.
    std::vector<int> route(int from, int to) const;

private:
    std::vector<std::string> ids_;
    std::unordered_map<std::string, int> index_;
    std::vector<double> lat_, lon_;
    std::vector<std::vector<Edge>> adj_;
    std::vector<double> dist_;   // nodeCount * nodeCount, row = source node
    std::vector<int> next_;      // first step on the shortest path from row to column, -1 if none
};
