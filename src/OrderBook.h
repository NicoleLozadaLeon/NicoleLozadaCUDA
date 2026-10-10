#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

#include "Order.h"

// Monitor: the orders waiting for a courier, in arrival order (FIFO), with a size limit.
class OrderBook {
public:
    explicit OrderBook(std::size_t maxPending) : maxPending_(maxPending) {}

    // Adds an order at the back. Returns false if the book is full; then the order is NOT
    // taken and the caller still owns it (it rejects it with queueFull).
    bool submit(Order&& order);
    // Puts at the FRONT an order the dispatcher could not place yet. Never refused.
    void putBack(Order&& order);
    // Blocks until an order is available, or returns std::nullopt once the book is closed and empty.
    std::optional<Order> take();
    // No more orders will arrive: wakes every thread blocked in take().
    void close();
    std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Order> orders_;
    const std::size_t maxPending_;
    bool closed_ = false;
};

