#pragma once
#include <condition_variable>

#include <mutex>
#include <string>
#include <utility>
#include "EventLog.h"
class Restaurant {
public:
    Restaurant(std::string id, std::string name, int node, int slots, EventLog& log)
        : id_(std::move(id)), name_(std::move(name)), node_(node), slots_(slots), log_(log) {}
    bool acquireSlot();
    void releaseSlot();
    int setPaused(bool on);
    void stop();
    const std::string& id() const { return id_; }
    const std::string& name() const { return name_; }
    int node() const { return node_; }
private:
    const std::string id_, name_;
    const int node_;
    const int slots_;
    EventLog& log_;
    std::mutex mutex_;
    std::condition_variable turn_;
    int busy_ = 0;
    unsigned long long nextTicket_ = 0;
    unsigned long long serving_ = 0;
    bool stopping_ = false;
    bool paused_ = false;
};
class SlotGuard {
public:
    explicit SlotGuard(Restaurant& restaurant) : restaurant_(restaurant) {}
    ~SlotGuard() { restaurant_.releaseSlot(); }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;
private:
    Restaurant& restaurant_;
};
