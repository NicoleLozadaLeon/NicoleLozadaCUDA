#pragma once
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <vector>
#include "Config.h"
#include "EventLog.h"
#include "Restaurant.h"
#include "SimClock.h"
class KitchenScheduler {
public:
    KitchenScheduler(const Config& cfg, const SimClock& clock, EventLog& log,
                     const std::vector<std::shared_ptr<Restaurant>>& restaurants);
    void run(std::promise<void> ready);
    void stop();
private:
    struct Plan {
        bool paused;
        long long nextMs;
    };
    bool sleepUntil(long long simMs);
    const SimClock& clock_;
    EventLog& log_;
    const std::vector<std::shared_ptr<Restaurant>>& restaurants_;
    const long long everyMs_;
    const long long pauseMs_;
    std::vector<Plan> plan_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
};
