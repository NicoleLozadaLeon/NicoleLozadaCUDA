#pragma once
#include <string>
#include <utility>

// An order exists in exactly one place at a time: the generator, the OrderBook,
// the dispatcher or one courier's bag. It can be moved, never copied.
struct Order {
    std::string id;
    int restaurant;          // index into Config::restaurants
    int deliveryNode;        // Graph node index of the customer
    long long createdAtMs;   // simulated ms
    long long prepMs;        // preparation time drawn by the generator

    Order(std::string id_, int restaurant_, int deliveryNode_, long long createdAtMs_, long long prepMs_)
        : id(std::move(id_)), restaurant(restaurant_), deliveryNode(deliveryNode_),
          createdAtMs(createdAtMs_), prepMs(prepMs_) {}

    Order(const Order&) = delete;
    Order& operator=(const Order&) = delete;
    Order(Order&&) noexcept = default;
    Order& operator=(Order&&) noexcept = default;
    ~Order() = default;
};

