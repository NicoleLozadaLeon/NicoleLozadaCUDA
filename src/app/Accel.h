#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>
#include "Argmin.h"
#include "Backlog.h"
#include "Eta.h"
#include "FleetSummary.h"
#include "Floyd.h"
#include "FoodWait.h"
#include "GpuWorker.h"
#include "MapGrid.h"
#include "Nearby.h"
#include "Scan.h"

class Accel {
public:
    Accel(bool wantGpu, int cpuThreads);
    ~Accel();
    Accel(const Accel&) = delete;
    Accel& operator=(const Accel&) = delete;

    bool gpu() const { return worker_ != nullptr; }
    bool device() const { return device_; }
    std::string backend() const;

    std::vector<int> allPairs(const floyd::Network& net);
    void keepTable(const std::vector<int>& dist, int n);
    std::future<argmin::Key> bestCourier(eta::Query q, eta::Params p, eta::Couriers couriers, std::vector<int> ids);

    std::vector<unsigned> deliveryHistogram(const std::vector<int>& deliveryMs, int widthMs, int bins);
    scan::Grouping groupByRestaurant(const std::vector<int>& restaurant, int restaurants);

    std::future<Grid> smooth(Grid in, int radius);
    std::future<Grid> coverage(Grid sources, int sweeps);

    // The second application of each concept, used once in the final summary (synchronous).
    std::vector<int> couriersNear(const nearby::Points& corners, const nearby::Points& couriers, int radiusM);
    std::vector<int> foodWaiting(const foodwait::Items& items, long long nowMs);
    fleetsum::Summary fleetSummary(const fleetsum::FleetData& fleet);
    backlog::Result orderQueue(const backlog::EventStream& events);
    Grid pendingDemand(int w, int h, const std::vector<int>& cells);

private:
    template <class Fn>
    auto onWorker(Fn fn) -> decltype(fn()) {
        return worker_->submit(std::move(fn)).get();
    }

    bool device_ = false;
    int threads_;
    std::unique_ptr<GpuWorker> worker_;
    std::unique_ptr<eta::DeviceTable> table_;
    std::vector<int> hostTable_;
};
