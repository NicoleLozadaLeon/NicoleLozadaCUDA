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
// One motorcycle. It is an ACTOR: its own thread runs run() for the whole simulation.
// It is also a MONITOR: every member below mutex_ is only touched while holding mutex_.
class Courier {
public:
    static constexpr long long kHandoverMs = 30000;   // simulated time to collect food at a slot
    static constexpr long long kRepairMs = 300000;    // simulated time a breakdown lasts
    enum class State { Idle, ToRestaurant, Carrying, Broken };
    struct BagItem {
        Order order;
        int restaurantNode;
        long long readyAtMs;   // createdAtMs + prepMs: the food is ready
        bool pickupStarted = false;
        bool pickedUp = false;
    };
    // A plain copy of the courier at one instant, for the dispatcher and the window.
    struct View {
        struct Item {
            int restaurantNode;
            int deliveryNode;
            long long readyAtMs;
            bool pickedUp;
        };
        int index = 0;
        State state = State::Idle;
        bool available = false;     // not broken, not stopping, room in the bag
        int legFrom = 0, legTo = 0; // equal when the courier stands at a node
        long long legStartMs = 0, legEndMs = 0;
        std::vector<Item> bag;
    };
    Courier(int index, const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
            EventLog& log, const std::vector<std::shared_ptr<Restaurant>>& restaurants, Fleet& fleet);
    void run(std::promise<void> ready);   // the body of this courier's thread
    void stop();                          // wakes the thread from any wait; it ends soon after
    // Called by the dispatcher. If the courier can still take the order, it is MOVED into the
    // bag and true is returned; otherwise `order` is untouched and the caller still owns it.
    bool offer(Order& order);
    View view() const;
    const std::string& id() const { return id_; }
    double metersPerSimMs() const { return metersPerSimMs_; }

    // Breakdown handover: moves EVERY order of `from` into `to`, or none of them.
    static bool transferAll(Courier& from, Courier& to, EventLog& log);
private:
    enum class Ride { Arrived, BrokeDown, Stuck, Stopped };
    struct Task {
        bool pickup;
        int restaurant;   // index, when pickup
        int node;         // where to ride
    };
    bool nextTask(Task& task);
    bool pickupAt(int restaurant);   // false = the simulation is stopping
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
    std::mt19937_64 rng_;   // used only by this courier's own thread: no lock needed
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::vector<BagItem> bag_;
    int node_;                // last node reached (written only by this courier's thread)
    int legFrom_, legTo_;
    long long legStartMs_ = 0, legEndMs_ = 0;
    State state_ = State::Idle;
    bool broken_ = false;
    bool stopping_ = false;
};