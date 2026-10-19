#pragma once
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SimClock.h"

class EventLog {
public:
    struct Summary {
        long long created = 0, delivered = 0, rejected = 0, pending = 0;
        long long reassigned = 0, breakdowns = 0;
        long long avgDeliveryMs = 0, maxDeliveryMs = 0;
        std::map<std::string, long long> deliveredPerCourier;
    };

    EventLog(const std::string& path, const SimClock& clock);

    long long orderCreated(const std::string& order, const std::string& restaurant);
    void registerCourier(const std::string& courier);

    void orderAssigned(const std::string& order, const std::string& courier);
    void pickupStarted(const std::string& order, const std::string& courier, const std::string& restaurant);
    void orderPickedUp(const std::string& order, const std::string& courier, const std::string& restaurant);
    void orderDelivered(const std::string& order, const std::string& courier);
    void orderRejected(const std::string& order, const std::string& reason);
    void courierBreakdown(const std::string& courier);
    void orderReassigned(const std::string& order, const std::string& from, const std::string& to);
    void simulationStopping();

    Summary summary() const;
    std::string summaryJson() const;

private:
    using Fields = std::vector<std::pair<std::string, std::string>>;
    long long emit_(const std::string& event, const Fields& fields);

    mutable std::mutex mutex_;
    std::ofstream out_;
    const SimClock& clock_;
    Summary counters_;
    std::unordered_map<std::string, long long> createdAt_;
    long long totalDeliveryMs_ = 0;
};
