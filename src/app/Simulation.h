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
// Owns every shared object and every thread. The member order is deliberate: members are
// built top to bottom and destroyed bottom to top, so everything a thread uses is built
// before the thread starts and destroyed only after it has been joined.
class Simulation {
public:
    Simulation(const Config& cfg, const Settings& settings, const std::string& logPath);
    ~Simulation();   // stop(): no thread is ever abandoned, even when an exception unwinds main
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    void start();           // starts every thread; returns once ALL of them are ready
    void stop();            // orderly shutdown; safe to call more than once
    bool finished() const;  // simulation.durationS reached
    // Read-only access for the window and main.
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
    Graph graph_;           // read-only once built
    SimClock clock_;        // starts after the graph is ready
    EventLog log_;          // innermost lock of the program
    OrderBook book_;
    std::vector<std::shared_ptr<Restaurant>> restaurants_;
    Fleet fleet_;
    EstimatePool pool_;     // its workers start here and are joined in stop()
    Dispatcher dispatcher_;
    OrderGenerator generator_;
    std::thread generatorThread_;
    std::thread dispatcherThread_;
    std::vector<std::thread> courierThreads_;
    bool started_ = false;
    bool stopped_ = false;
};