#pragma once
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include "Config.h"
#include "EventLog.h"
#include "Graph.h"
#include "OrderBook.h"
#include "Restaurant.h"
#include "SimClock.h"
// The order generator thread: creates orders at random moments, alone or in bursts, for the
// whole run. Every random decision comes from ONE generator seeded with orders.seed, used only
// by this thread, so a seed always gives the same sequence of orders.
class OrderGenerator {
public:
    OrderGenerator(const Config& cfg, const Graph& graph, const SimClock& clock, EventLog& log, OrderBook& book,
                   const std::vector<std::shared_ptr<Restaurant>>& restaurants);
    void run(std::promise<void> ready);   // the body of the generator thread
    void stop();                          // wakes it from its timed wait; it ends at once
private:
    bool sleepUntil(long long simMs);     // false = stopping
    void createOrder();
    const Config& cfg_;
    const Graph& graph_;
    const SimClock& clock_;
    EventLog& log_;
    OrderBook& book_;
    const std::vector<std::shared_ptr<Restaurant>>& restaurants_;
    std::mt19937_64 rng_;
    long long nextId_ = 0;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
};