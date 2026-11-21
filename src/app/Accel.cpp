#include "Accel.h"
#include <stdexcept>
#include <utility>
#include "Conv.h"
#include "Gpu.h"
#include "Histogram.h"
#include "Hotspots.h"
#include "Stencil.h"

namespace {
template <class T>
std::future<T> ready(T value) {
    std::promise<T> p;
    p.set_value(std::move(value));
    return p.get_future();
}
}

Accel::Accel(bool wantGpu, int cpuThreads) : threads_(cpuThreads) {
    if (!wantGpu) return;
    worker_ = std::make_unique<GpuWorker>();
    device_ = worker_->submit([] { return gpu::compiled() && gpu::start(); }).get();
}

Accel::~Accel() {
    if (!worker_) return;
    worker_->submit([this] { table_.reset(); }).get();
    worker_->shutdown();
}

std::string Accel::backend() const {
    if (!worker_) return "cpu (" + std::to_string(threads_) + " threads)";
    if (!device_) return "gpu worker running the CPU versions (no CUDA device)";
    return "gpu (" + worker_->submit([] { return gpu::deviceName(); }).get() + ")";
}

std::vector<int> Accel::allPairs(const floyd::Network& net) {
    if (!worker_) return floyd::dijkstraMulti(net, threads_);
    return onWorker([this, &net] {
        if (!device_) return floyd::dijkstraMulti(net, threads_);
        std::vector<int> dist;
        floyd::floydGpu(net, dist);
        return dist;
    });
}

void Accel::keepTable(const std::vector<int>& dist, int n) {
    if (!worker_) return;
    onWorker([this, &dist, n] {
        if (device_) table_ = std::make_unique<eta::DeviceTable>(dist, n);
        else hostTable_ = dist;
    });
}

std::future<argmin::Key> Accel::bestCourier(eta::Query q, eta::Params p, eta::Couriers couriers,
                                            std::vector<int> ids) {
    if (!worker_) throw std::logic_error("bestCourier needs the GPU worker");
    return worker_->submit([this, q, p, couriers = std::move(couriers), ids = std::move(ids)] {
        if (device_) return argmin::bestCourierGpu(*table_, p, q, couriers, ids);
        const std::vector<int> etas = eta::cpuOne(hostTable_, p, {q}, couriers);
        return argmin::cpuOne(etas, 1, static_cast<int>(ids.size()), ids).front();
    });
}

std::vector<unsigned> Accel::deliveryHistogram(const std::vector<int>& deliveryMs, int widthMs, int bins) {
    if (!worker_) return hist::byWidthMulti(deliveryMs, widthMs, bins, threads_);
    return onWorker([&] {
        if (!device_) return hist::byWidthMulti(deliveryMs, widthMs, bins, threads_);
        std::vector<unsigned> out;
        hist::byWidthGpu(deliveryMs, widthMs, bins, out);
        return out;
    });
}

scan::Grouping Accel::groupByRestaurant(const std::vector<int>& restaurant, int restaurants) {
    if (!worker_) return scan::groupMulti(restaurant, restaurants, threads_);
    return onWorker([&] {
        if (!device_) return scan::groupMulti(restaurant, restaurants, threads_);
        scan::Grouping out;
        scan::groupGpu(restaurant, restaurants, out);
        return out;
    });
}

std::future<Grid> Accel::smooth(Grid in, int radius) {
    if (!worker_) return ready(conv::cpuMulti(in, conv::gaussian(radius), radius, threads_));
    return worker_->submit([this, in = std::move(in), radius] {
        if (!device_) return conv::cpuMulti(in, conv::gaussian(radius), radius, threads_);
        Grid out;
        conv::gpu(in, conv::gaussian(radius), radius, out);
        return out;
    });
}

std::future<Grid> Accel::coverage(Grid sources, int sweeps) {
    if (!worker_) return ready(stencil::cpuMulti(sources, sweeps, threads_));
    return worker_->submit([this, sources = std::move(sources), sweeps] {
        if (!device_) return stencil::cpuMulti(sources, sweeps, threads_);
        Grid out;
        stencil::gpu(sources, sweeps, out);
        return out;
    });
}

std::vector<int> Accel::couriersNear(const nearby::Points& corners, const nearby::Points& couriers, int radiusM) {
    if (!worker_) return nearby::cpuMulti(corners, couriers, radiusM, threads_);
    return onWorker([&] {
        if (!device_) return nearby::cpuMulti(corners, couriers, radiusM, threads_);
        const nearby::DeviceCorners onDevice(corners);
        std::vector<int> out;
        nearby::gpuTiled(onDevice, couriers, radiusM, out);
        return out;
    });
}

std::vector<int> Accel::foodWaiting(const foodwait::Items& items, long long nowMs) {
    if (!worker_) return foodwait::cpuMulti(items, nowMs, threads_);
    return onWorker([&] {
        if (!device_) return foodwait::cpuMulti(items, nowMs, threads_);
        std::vector<int> out;
        foodwait::gpuSoA(items, nowMs, out);
        return out;
    });
}

fleetsum::Summary Accel::fleetSummary(const fleetsum::FleetData& fleet) {
    if (!worker_) return fleetsum::cpuMulti(fleet, threads_);
    return onWorker([&] {
        if (!device_) return fleetsum::cpuMulti(fleet, threads_);
        fleetsum::Summary out = fleetsum::identity();
        fleetsum::gpuBranchFree(fleet, out);
        return out;
    });
}

backlog::Result Accel::orderQueue(const backlog::EventStream& events) {
    if (!worker_) return backlog::cpuMulti(events, threads_);
    return onWorker([&] {
        if (!device_) return backlog::cpuMulti(events, threads_);
        backlog::Result out;
        backlog::gpuScan(events, out);
        return out;
    });
}

Grid Accel::pendingDemand(int w, int h, const std::vector<int>& cells) {
    const std::vector<float> filter = hotspots::gaussian();
    if (!worker_) return hotspots::cpuMulti(w, h, cells, filter, threads_);
    return onWorker([&] {
        if (!device_) return hotspots::cpuMulti(w, h, cells, filter, threads_);
        Grid out;
        hotspots::gpuDirect(w, h, cells, filter, out);
        return out;
    });
}
