#pragma once
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "Config.h"
#include "Courier.h"
#include "EstimatePool.h"
#include "EventLog.h"
#include "Fleet.h"
#include "Graph.h"
#include "OrderBook.h"
#include "Settings.h"
#include "SimClock.h"
long long estimateDeliveryMs(const Graph& graph, const Courier::View& courier, int restaurantNode,
                             int deliveryNode, long long readyAtMs, long long nowMs, double metersPerSimMs);
class Dispatcher {
public:
    Dispatcher(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
               EventLog& log, OrderBook& book, Fleet& fleet, EstimatePool& pool,
               const std::vector<std::shared_ptr<Restaurant>>& restaurants);
    void run(std::promise<void> ready);
    std::string statsLine() const;
    long long late() const { return late_.load(std::memory_order_relaxed); }
    long long failed() const { return failed_.load(std::memory_order_relaxed); }
private:
    bool dispatch(Order order);
    std::vector<std::pair<long long, int>> rankCouriers(const Order& order);
    std::future<long long> launch(std::function<long long()> estimate);
    const Config& cfg_;
    const Settings& settings_;
    const Graph& graph_;
    const SimClock& clock_;
    EventLog& log_;
    OrderBook& book_;
    Fleet& fleet_;
    EstimatePool& pool_;
    const std::vector<std::shared_ptr<Restaurant>>& restaurants_;
    const double metersPerSimMs_;
    std::mt19937_64 rng_;
    std::atomic<long long> rankings_{0}, estimates_{0}, late_{0}, failed_{0};
    std::atomic<long long> totalRankUs_{0}, maxRankUs_{0};
};
