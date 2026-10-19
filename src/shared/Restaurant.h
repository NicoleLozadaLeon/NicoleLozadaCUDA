#pragma once
#include <condition_variable>

#include <mutex>
#include <string>
#include <utility>
// Monitor: the pickup slots of one restaurant. At most `slots` couriers collect food at the
// same time, and waiting couriers are served strictly in arrival order (ticket queue).
class Restaurant {
public:
    Restaurant(std::string id, std::string name, int node, int slots)
        : id_(std::move(id)), name_(std::move(name)), node_(node), slots_(slots) {}
    // Blocks, without CPU, until it is this courier's turn AND a slot is free.
    // Returns false if the simulation is stopping (then no slot is held).
    bool acquireSlot();
    void releaseSlot();
    void stop();   // wakes every waiting courier
    // Constant after construction: read without the lock.
    const std::string& id() const { return id_; }
    const std::string& name() const { return name_; }
    int node() const { return node_; }
private:
    const std::string id_, name_;
    const int node_;
    const int slots_;
    std::mutex mutex_;
    std::condition_variable turn_;
    int busy_ = 0;                        // couriers collecting right now
    unsigned long long nextTicket_ = 0;   // ticket for the next courier that arrives
    unsigned long long serving_ = 0;      // smallest ticket not admitted yet
    bool stopping_ = false;
};
// RAII: releases the slot on every way out of the scope, exceptions and early returns included.
class SlotGuard {
public:
    explicit SlotGuard(Restaurant& restaurant) : restaurant_(restaurant) {}
    ~SlotGuard() { restaurant_.releaseSlot(); }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;
private:
    Restaurant& restaurant_;
};