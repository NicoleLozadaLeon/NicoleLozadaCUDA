#pragma once
#include <atomic>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "Config.h"

class Accel;

class Graph {
public:
    static constexpr double kUnreachable = std::numeric_limits<double>::infinity();

    struct Edge {
        int to;
        double meters;
        int street;
    };
    explicit Graph(const Config& cfg, int tasks = 0, int grain = 4, Accel* accel = nullptr);

    int nodeCount() const;
    int indexOf(const std::string& id) const;
    const std::string& idOf(int node) const;
    double lat(int node) const;
    double lon(int node) const;
    const std::vector<Edge>& edgesFrom(int node) const;

    double distance(int from, int to) const;
    bool reachable(int from, int to) const;
    std::vector<int> route(int from, int to) const;

    double buildMs() const { return buildMs_; }
    int tasksUsed() const { return tasksUsed_; }
    int grain() const { return grain_; }
    bool builtOnGpu() const { return builtOnGpu_; }
    const std::vector<int>& distDm() const { return distDm_; }

private:
    static int edgeDm(const Edge& e);
    void buildWithAccel(Accel& accel);
    void firstHops(int begin, int end);
    void shortestPathsFrom(int source);
    void claimSources(std::atomic<int>& next, int grain);
    void sourceRange(int begin, int end);
    std::vector<std::string> ids_;
    std::unordered_map<std::string, int> index_;
    std::vector<double> lat_, lon_;
    std::vector<std::vector<Edge>> adj_;
    std::vector<double> dist_;
    std::vector<int> next_;
    std::vector<int> distDm_;
    bool builtOnGpu_ = false;
    double buildMs_ = 0.0;
    int tasksUsed_ = 1;
    int grain_ = 0;
};
