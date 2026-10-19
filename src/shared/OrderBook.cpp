#include "OrderBook.h"

#include <utility>

bool OrderBook::submit(Order&& order) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (orders_.size() >= maxPending_) return false;
        orders_.push_back(std::move(order));
    }
    ready_.notify_one();
    return true;
}

std::optional<Order> OrderBook::take() {
    std::unique_lock<std::mutex> lock(mutex_);

    ready_.wait(lock, [this] { return !orders_.empty() || closed_; });
    if (closed_) return std::nullopt;
    Order order = std::move(orders_.front());
    orders_.pop_front();
    return order;

}

void OrderBook::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    ready_.notify_all();
}

std::size_t OrderBook::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return orders_.size();
}

std::vector<int> OrderBook::deliveryNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> nodes;
    nodes.reserve(orders_.size());
    for (const Order& order : orders_) nodes.push_back(order.deliveryNode);
    return nodes;
}
