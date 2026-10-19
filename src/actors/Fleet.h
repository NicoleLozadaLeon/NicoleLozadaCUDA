#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include "Courier.h"
class Fleet {
public:
    Fleet(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock, EventLog& log,
          const std::vector<std::shared_ptr<Restaurant>>& restaurants);
    const std::vector<std::shared_ptr<Courier>>& couriers() const { return couriers_; }
    unsigned long long version() const;
    void notifyChanged();
    bool waitForChange(unsigned long long seen, SimClock::Clock::time_point deadline);
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
