#include "OrderBook.h"

#include <utility>

bool OrderBook::submit(Order&& order) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (orders_.size() >= maxPending_) return false;   // full: the caller rejects it (queueFull)
        orders_.push_back(std::move(order));
    }
    ready_.notify_one();   // after unlocking: the woken thread does not wake up just to block again
    return true;
}

std::size_t OrderBook::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return orders_.size();
}

// TODO 1: putBack
// TODO 2: take
// TODO 3: close
