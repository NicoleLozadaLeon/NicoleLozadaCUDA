#include "KitchenScheduler.h"
#include <cstddef>
KitchenScheduler::KitchenScheduler(const Config& cfg, const SimClock& clock, EventLog& log,
                                   const std::vector<std::shared_ptr<Restaurant>>& restaurants)
    : clock_(clock),
      log_(log),
      restaurants_(restaurants),
      everyMs_(cfg.kitchen.pauseEveryMs),
      pauseMs_(cfg.kitchen.pauseMs) {
    const long long count = static_cast<long long>(restaurants.size());
    for (long long i = 0; i < count; ++i) plan_.push_back(Plan{false, i * everyMs_ / count});
}
void KitchenScheduler::run(std::promise<void> ready) {
    ready.set_value();
    while (true) {
        std::size_t i = 0;
        for (std::size_t k = 1; k < plan_.size(); ++k)
            if (plan_[k].nextMs < plan_[i].nextMs) i = k;
        const long long t = plan_[i].nextMs;
        if (!sleepUntil(t)) return;
        Restaurant& restaurant = *restaurants_[i];
        if (!plan_[i].paused) {
            const int queued = restaurant.setPaused(true);
            log_.restaurantPaused(restaurant.id(), t);
            log_.pickupsDelayed(queued);
            plan_[i].paused = true;
            plan_[i].nextMs = t + pauseMs_;
        } else {
            log_.restaurantResumed(restaurant.id(), t);
            restaurant.setPaused(false);
            plan_[i].paused = false;
            plan_[i].nextMs = t - pauseMs_ + everyMs_;
        }
    }
}
bool KitchenScheduler::sleepUntil(long long simMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    return !wake_.wait_until(lock, clock_.toReal(simMs), [this] { return stopping_; });
}
void KitchenScheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
}
