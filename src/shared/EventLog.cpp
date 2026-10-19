#include "EventLog.h"

#include <algorithm>
#include <stdexcept>

#include <nlohmann/json.hpp>

EventLog::EventLog(const std::string& path, const SimClock& clock) : out_(path), clock_(clock) {
    if (!out_) throw std::runtime_error("cannot open the event log file: " + path);
}

long long EventLog::emit_(const std::string& event, const Fields& fields) {
    nlohmann::ordered_json line;   // keeps "t" and "event" first, as in the brief
    const long long t = clock_.nowMs();   // read inside the lock: t never goes backwards in the file
    line["t"] = t;
    line["event"] = event;
    for (const auto& f : fields) line[f.first] = f.second;
    out_ << line.dump() << '\n';
    out_.flush();
    return t;
}

long long EventLog::orderCreated(const std::string& order, const std::string& restaurant) {
    std::lock_guard<std::mutex> lock(mutex_);
    const long long t = emit_("orderCreated", {{"order", order}, {"restaurant", restaurant}});
    createdAt_[order] = t;
    ++counters_.created;
    return t;
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


void EventLog::registerCourier(const std::string& courier) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.deliveredPerCourier.emplace(courier, 0);
}
void EventLog::orderAssigned(const std::string& order, const std::string& courier) {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("orderAssigned", {{"order", order}, {"courier", courier}});
}

void EventLog::pickupStarted(const std::string& order, const std::string& courier,
                             const std::string& restaurant) {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("pickupStarted", {{"order", order}, {"courier", courier}, {"restaurant", restaurant}});
}

void EventLog::orderPickedUp(const std::string& order, const std::string& courier,
                             const std::string& restaurant) {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("orderPickedUp", {{"order", order}, {"courier", courier}, {"restaurant", restaurant}});
}

void EventLog::orderRejected(const std::string& order, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("orderRejected", {{"order", order}, {"reason", reason}});
    ++counters_.rejected;
}

void EventLog::courierBreakdown(const std::string& courier) {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("courierBreakdown", {{"courier", courier}});
    ++counters_.breakdowns;
}

void EventLog::orderReassigned(const std::string& order, const std::string& from, const std::string& to) {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("orderReassigned", {{"order", order}, {"from", from}, {"to", to}});
    ++counters_.reassigned;
}

void EventLog::simulationStopping() {
    std::lock_guard<std::mutex> lock(mutex_);
    emit_("simulationStopping", {});
}

EventLog::Summary EventLog::summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Summary s = counters_;
    s.pending = s.created - s.delivered - s.rejected;
    s.avgDeliveryMs = s.delivered > 0 ? totalDeliveryMs_ / s.delivered : 0;
    return s;
}

std::string EventLog::summaryJson() const {
    const Summary s = summary();
    nlohmann::ordered_json j;   // keys in the order of the brief
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
