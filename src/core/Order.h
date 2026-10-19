#pragma once
#include <string>
#include <utility>

struct Order {
    std::string id;
    int restaurant;
    int deliveryNode;
    long long createdAtMs;
    long long prepMs;

    Order(std::string id_, int restaurant_, int deliveryNode_, long long createdAtMs_, long long prepMs_)
        : id(std::move(id_)), restaurant(restaurant_), deliveryNode(deliveryNode_),
          createdAtMs(createdAtMs_), prepMs(prepMs_) {}

    Order(const Order&) = delete;
    Order& operator=(const Order&) = delete;
    Order(Order&&) noexcept = default;
    Order& operator=(Order&&) noexcept = default;
    ~Order() = default;
};
