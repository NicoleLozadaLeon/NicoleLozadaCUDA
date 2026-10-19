#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>
#include "Order.h"

class OrderBook {
public:
    explicit OrderBook(std::size_t maxPending) : maxPending_(maxPending) {}

    bool submit(Order&& order);

    std::optional<Order> take();
    void close();
    std::size_t size() const;
    std::vector<int> deliveryNodes() const;
private:
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Order> orders_;
    const std::size_t maxPending_;
    bool closed_ = false;
};
