#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include "Courier.h"
// Owns the couriers and tells waiting threads when "something changed in the fleet"
// (a bag got room, a courier was repaired). The couriers_ vector is filled in the
// constructor and never changed again, so it is read without a lock; the change counter
// is a small monitor of its own.
class Fleet {
public:
    Fleet(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock, EventLog& log,
          const std::vector<std::shared_ptr<Restaurant>>& restaurants);
    const std::vector<std::shared_ptr<Courier>>& couriers() const { return couriers_; }
    unsigned long long version() const;
    void notifyChanged();
    // Waits until the version differs from `seen`, the deadline passes or the fleet stops.
    // Returns false if the fleet is stopping.
    bool waitForChange(unsigned long long seen, SimClock::Clock::time_point deadline);
    // Hands every order of a broken courier to ONE other courier, closest first.
    bool handOver(Courier& from);
    void stop();
    bool stopping() const;
private:
    const Graph& graph_;
    EventLog& log_;
    std::vector<std::shared_ptr<Courier>> couriers_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    unsigned long long version_ = 0;
    bool stopping_ = false;
};