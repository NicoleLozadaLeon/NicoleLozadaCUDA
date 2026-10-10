#include "EventLog.h"

#include <algorithm>
#include <stdexcept>

#include <nlohmann/json.hpp>

EventLog::EventLog(const std::string& path, const SimClock& clock) : out_(path), clock_(clock) {
    if (!out_) throw std::runtime_error("cannot open the event log file: " + path);
}

long long EventLog::emit_(const std::string& event, const Fields& fields) {
    nlohmann::json line;
    const long long t = clock_.nowMs();   // read inside the lock: t never goes backwards in the file
    line["t"] = t;
    line["event"] = event;
    for (const auto& f : fields) line[f.first] = f.second;
    out_ << line.dump() << '\n';
    out_.flush();
    return t;
}

void EventLog::orderCreated(const std::string& order, const std::string& restaurant) {
    std::lock_guard<std::mutex> lock(mutex_);
    const long long t = emit_("orderCreated", {{"order", order}, {"restaurant", restaurant}});
    createdAt_[order] = t;
    ++counters_.created;
}

void EventLog::orderDelivered(const std::string& order, const std::string& courier) {
    std::lock_guard<std::mutex> lock(mutex_);
    const long long t = emit_("orderDelivered", {{"order", order}, {"courier", courier}});
    ++counters_.delivered;
    ++counters_.deliveredPerCourier[courier];
    const auto it = createdAt_.find(order);
    if (it != createdAt_.end()) {
        const long long duration = t - it->second;
        totalDeliveryMs_ += duration;
        counters_.maxDeliveryMs = std::max(counters_.maxDeliveryMs, duration);
    }
}

// TODO A: orderAssigned(order, courier)                      fields: order, courier
// TODO B: pickupStarted(order, courier, restaurant)          fields: order, courier, restaurant
// TODO C: orderPickedUp(order, courier, restaurant)          fields: order, courier, restaurant
// TODO D: orderRejected(order, reason)                       fields: order, reason; ++rejected
// TODO E: courierBreakdown(courier)                          fields: courier; ++breakdowns
// TODO F: orderReassigned(order, from, to)                   fields: order, from, to; ++reassigned
// TODO G: simulationStopping()                               no fields

EventLog::Summary EventLog::summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Summary s = counters_;
    s.pending = s.created - s.delivered - s.rejected;
    s.avgDeliveryMs = s.delivered > 0 ? totalDeliveryMs_ / s.delivered : 0;
    return s;
}

std::string EventLog::summaryJson() const {
    const Summary s = summary();
    nlohmann::json j;
    j["created"] = s.created;
    j["delivered"] = s.delivered;
    j["rejected"] = s.rejected;
    j["pending"] = s.pending;
    j["reassigned"] = s.reassigned;
    j["breakdowns"] = s.breakdowns;
    j["avgDeliveryMs"] = s.avgDeliveryMs;
    j["maxDeliveryMs"] = s.maxDeliveryMs;
    j["deliveredPerCourier"] = s.deliveredPerCourier;
    return j.dump();
}
