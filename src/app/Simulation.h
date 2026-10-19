#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "Config.h"
#include "Dispatcher.h"
#include "EstimatePool.h"
#include "EventLog.h"
#include "Fleet.h"
#include "Graph.h"
#include "OrderBook.h"
#include "OrderGenerator.h"
#include "Restaurant.h"
#include "Settings.h"
#include "SimClock.h"
class Simulation {
public:
    Simulation(const Config& cfg, const Settings& settings, const std::string& logPath);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    void start();
    void stop();
    bool finished() const;
    const Graph& graph() const { return graph_; }
    const SimClock& clock() const { return clock_; }
    const EventLog& log() const { return log_; }
    const OrderBook& book() const { return book_; }
    const Fleet& fleet() const { return fleet_; }
    const Dispatcher& dispatcher() const { return dispatcher_; }
    const std::vector<std::shared_ptr<Restaurant>>& restaurants() const { return restaurants_; }
    std::string graphReport() const;
private:
    const Config& cfg_;
    Graph graph_;
    SimClock clock_;
    EventLog log_;
    OrderBook book_;
    std::vector<std::shared_ptr<Restaurant>> restaurants_;
    Fleet fleet_;
    EstimatePool pool_;
    Dispatcher dispatcher_;
    OrderGenerator generator_;
    std::thread generatorThread_;
    std::thread dispatcherThread_;
    std::vector<std::thread> courierThreads_;
    bool started_ = false;
    bool stopped_ = false;
};
