#include "Restaurant.h"
bool Restaurant::acquireSlot() {
    std::unique_lock<std::mutex> lock(mutex_);
    const unsigned long long ticket = nextTicket_++;
    turn_.wait(lock, [&] { return stopping_ || (ticket == serving_ && busy_ < slots_); });
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
