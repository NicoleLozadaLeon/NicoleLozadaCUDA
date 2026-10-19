#pragma once
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>
#include "Config.h"
#include "EventLog.h"
#include "Graph.h"
#include "Order.h"
#include "Restaurant.h"
#include "Settings.h"
#include "SimClock.h"
class Fleet;
class Courier {
public:
    static constexpr long long kHandoverMs = 30000;
    static constexpr long long kRepairMs = 300000;
    enum class State { Idle, ToRestaurant, Carrying, Broken };
    struct BagItem {
        Order order;
        int restaurantNode;
        long long readyAtMs;
        bool pickupStarted = false;
        bool pickedUp = false;
    };
    struct View {
        struct Item {
            int restaurantNode;
            int deliveryNode;
            long long readyAtMs;
            bool pickedUp;
        };
        int index = 0;
        State state = State::Idle;
        bool available = false;
        int legFrom = 0, legTo = 0;
        long long legStartMs = 0, legEndMs = 0;
        std::vector<Item> bag;
    };
    Courier(int index, const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
            EventLog& log, const std::vector<std::shared_ptr<Restaurant>>& restaurants, Fleet& fleet);
    void run(std::promise<void> ready);
    void stop();
    bool offer(Order& order);
    View view() const;
    const std::string& id() const { return id_; }
    double metersPerSimMs() const { return metersPerSimMs_; }

    static bool transferAll(Courier& from, Courier& to, EventLog& log);
private:
    enum class Ride { Arrived, BrokeDown, Stuck, Stopped };
    struct Task {
        bool pickup;
        int restaurant;
        int node;
    };
    bool nextTask(Task& task);
    bool pickupAt(int restaurant);
    bool deliverAt(int node);
    Ride rideTo(int target, State state);
    bool breakDown();
    bool waitForStop();
    bool sleepUntil(long long simMs);
    long long legMs(double meters) const;
    const int index_;
    const std::string id_;
    const std::size_t capacity_;
    const double metersPerSimMs_;
    const double breakdownProbability_;
    const Settings& settings_;
    const Graph& graph_;
    const SimClock& clock_;
    EventLog& log_;
    const std::vector<std::shared_ptr<Restaurant>>& restaurants_;
    Fleet& fleet_;
    std::mt19937_64 rng_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<BagItem> bag_;
    int node_;
    int legFrom_, legTo_;
    long long legStartMs_ = 0, legEndMs_ = 0;
    State state_ = State::Idle;
    bool broken_ = false;
    bool stopping_ = false;
};
