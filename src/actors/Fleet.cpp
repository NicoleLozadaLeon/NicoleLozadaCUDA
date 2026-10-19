#include "Fleet.h"
#include <algorithm>
#include <utility>
Fleet::Fleet(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock, EventLog& log,
             const std::vector<std::shared_ptr<Restaurant>>& restaurants)
    : graph_(graph), log_(log) {
    for (int i = 0; i < cfg.fleet.couriers; ++i) {
        couriers_.push_back(std::make_shared<Courier>(i, cfg, settings, graph, clock, log, restaurants, *this));
        log.registerCourier(couriers_.back()->id());
    }
}
unsigned long long Fleet::version() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return version_;
}
void Fleet::notifyChanged() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++version_;
    }
    changed_.notify_all();
}
bool Fleet::waitForChange(unsigned long long seen, SimClock::Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait_until(lock, deadline, [&] { return stopping_ || version_ != seen; });
    return !stopping_;
}
bool Fleet::handOver(Courier& from) {
    const Courier::View me = from.view();
    if (me.bag.empty()) return true;
    std::vector<std::pair<double, std::shared_ptr<Courier>>> receivers;
    for (const std::shared_ptr<Courier>& c : couriers_) {
        if (c.get() == &from) continue;
        const Courier::View v = c->view();
        if (v.available) receivers.push_back({graph_.distance(v.legTo, me.legTo), c});
    }
    std::sort(receivers.begin(), receivers.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& receiver : receivers)
        if (Courier::transferAll(from, *receiver.second, log_)) return true;
    return false;
}
void Fleet::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
    for (const std::shared_ptr<Courier>& c : couriers_) c->stop();
}
bool Fleet::stopping() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopping_;
}
