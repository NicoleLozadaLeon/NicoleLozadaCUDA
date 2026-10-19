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
// Estimated simulated ms until `courier` would deliver a new order (restaurant node,
// delivery node, food ready time). A pure function of its arguments: it reads the const
// Graph and its own copies, nothing shared and mutable. Throws if a node cannot be reached.
long long estimateDeliveryMs(const Graph& graph, const Courier::View& courier, int restaurantNode,
                             int deliveryNode, long long readyAtMs, long long nowMs, double metersPerSimMs);
// The dispatcher thread: takes orders from the book (FIFO) and gives each one to the courier
// with the best estimate. The estimates of the candidates run at the same time, off this thread.
class Dispatcher {
public:
    Dispatcher(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
               EventLog& log, OrderBook& book, Fleet& fleet, EstimatePool& pool,
               const std::vector<std::shared_ptr<Restaurant>>& restaurants);
    void run(std::promise<void> ready);   // the body of the dispatcher thread
    std::string statsLine() const;        // for stderr and the experiments
    // Statistics: written by the dispatcher thread, read live by the window. Relaxed atomics:
    // each counter only needs to be indivisible, nobody uses one to order other memory.
    long long late() const { return late_.load(std::memory_order_relaxed); }
    long long failed() const { return failed_.load(std::memory_order_relaxed); }
private:
    bool dispatch(Order order);   // by value: the order is moved in; false = stopping
    std::vector<std::pair<long long, int>> rankCouriers(const Order& order);   // (eta, courier index), best first
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
    std::mt19937_64 rng_;   // fault injection only; used by this thread only
    std::atomic<long long> rankings_{0}, estimates_{0}, late_{0}, failed_{0};
    std::atomic<long long> totalRankUs_{0}, maxRankUs_{0};
};