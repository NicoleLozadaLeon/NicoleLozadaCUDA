#include "Restaurant.h"
bool Restaurant::acquireSlot() {
    std::unique_lock<std::mutex> lock(mutex_);
    const unsigned long long ticket = nextTicket_++;
    if (paused_) log_.pickupsDelayed(1);
    turn_.wait(lock, [&] { return stopping_ || (!paused_ && ticket == serving_ && busy_ < slots_); });
    if (stopping_) return false;
    ++serving_;
    ++busy_;
    lock.unlock();
    turn_.notify_all();
    return true;
}
void Restaurant::releaseSlot() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        --busy_;
    }
    turn_.notify_all();
}
void Restaurant::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    turn_.notify_all();
}
int Restaurant::setPaused(bool on) {
    int queued = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (on) queued = static_cast<int>(nextTicket_ - serving_);
        paused_ = on;
    }
    if (!on) turn_.notify_all();
    return queued;
}
