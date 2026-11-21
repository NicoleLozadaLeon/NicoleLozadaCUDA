#include "Dispatcher.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
namespace {
long long travelMs(const Graph& graph, int courier, int from, int to, double metersPerSimMs) {
    const double meters = graph.distance(from, to);
    if (meters == Graph::kUnreachable)
        throw std::runtime_error("c" + std::to_string(courier) + " cannot reach node " + graph.idOf(to));
    return static_cast<long long>(meters / metersPerSimMs);
}
}
std::pair<long long, int> freeAfterBag(const Graph& graph, const Courier::View& courier, long long nowMs,
                                       double metersPerSimMs) {
    long long t = std::max(nowMs, courier.legEndMs);
    int at = courier.legTo;
    for (const Courier::View::Item& item : courier.bag) {
        if (item.pickedUp) continue;
        t = std::max(t + travelMs(graph, courier.index, at, item.restaurantNode, metersPerSimMs), item.readyAtMs) +
            Courier::kHandoverMs;
        at = item.restaurantNode;
    }
    for (const Courier::View::Item& item : courier.bag) {
        t += travelMs(graph, courier.index, at, item.deliveryNode, metersPerSimMs);
        at = item.deliveryNode;
    }
    return {t, at};
}
long long estimateDeliveryMs(const Graph& graph, const Courier::View& courier, int restaurantNode,
                             int deliveryNode, long long readyAtMs, long long nowMs, double metersPerSimMs) {
    auto [t, at] = freeAfterBag(graph, courier, nowMs, metersPerSimMs);
    t = std::max(t + travelMs(graph, courier.index, at, restaurantNode, metersPerSimMs), readyAtMs) +
        Courier::kHandoverMs;
    t += travelMs(graph, courier.index, restaurantNode, deliveryNode, metersPerSimMs);
    return t - nowMs;
}
Dispatcher::Dispatcher(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
                       EventLog& log, OrderBook& book, Fleet& fleet, EstimatePool& pool, Accel& accel,
                       const std::vector<std::shared_ptr<Restaurant>>& restaurants)
    : cfg_(cfg),
      settings_(settings),
      graph_(graph),
      clock_(clock),
      log_(log),
      book_(book),
      fleet_(fleet),
      pool_(pool),
      accel_(accel),
      restaurants_(restaurants),
      metersPerSimMs_(cfg.fleet.speedKmh / 3600.0),
      metersPerHour_(static_cast<long long>(cfg.fleet.speedKmh * 1000.0 + 0.5)),
      rng_(static_cast<std::uint64_t>(cfg.orders.seed) + 7u) {}
void Dispatcher::run(std::promise<void> ready) {
    ready.set_value();
    while (std::optional<Order> order = book_.take()) {
        if (!dispatch(std::move(*order))) break;
    }
}
bool Dispatcher::dispatch(Order order) {
    const long long acceptDeadline = order.createdAtMs + cfg_.dispatch.acceptTimeoutMs;
    while (true) {
        if (fleet_.stopping()) return false;
        const unsigned long long seen = fleet_.version();
        for (const std::pair<long long, int>& candidate : rankCouriers(order)) {
            if (fleet_.couriers()[static_cast<std::size_t>(candidate.second)]->offer(order)) return true;
        }
        if (clock_.nowMs() >= acceptDeadline) {
            log_.orderRejected(order.id, "noCourier");
            return true;
        }
        if (!fleet_.waitForChange(seen, clock_.toReal(acceptDeadline))) return false;

    }
}
std::future<long long> Dispatcher::launch(std::function<long long()> estimate) {
    switch (settings_.estimates) {
        case Settings::Estimates::Async:
            return std::async(std::launch::async, std::move(estimate));
        case Settings::Estimates::Deferred:
            return std::async(std::launch::deferred, std::move(estimate));
        case Settings::Estimates::Pool:
            break;
    }
    return pool_.submit(std::move(estimate));
}
std::vector<std::pair<long long, int>> Dispatcher::rankCouriers(const Order& order) {
    if (accel_.gpu()) return rankOnGpu(order);
    const auto started = std::chrono::steady_clock::now();
    const int restaurantNode = restaurants_[static_cast<std::size_t>(order.restaurant)]->node();
    const long long readyAt = order.createdAtMs + order.prepMs;
    const long long now = clock_.nowMs();
    std::vector<Courier::View> views;
    for (const std::shared_ptr<Courier>& courier : fleet_.couriers()) {
        Courier::View v = courier->view();
        if (v.available) views.push_back(std::move(v));
    }
    const std::size_t k = std::min(views.size(), static_cast<std::size_t>(settings_.candidates));
    std::partial_sort(views.begin(), views.begin() + static_cast<std::ptrdiff_t>(k), views.end(),
                      [&](const Courier::View& a, const Courier::View& b) {
                          return graph_.distance(a.legTo, restaurantNode) < graph_.distance(b.legTo, restaurantNode);
                      });
    views.resize(k);
    std::vector<std::pair<long long, int>> ranked;
    {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::vector<std::future<long long>> futures;
        futures.reserve(views.size());
        for (std::size_t i = 0; i < views.size(); ++i) {
            const bool inject = unit(rng_) < settings_.failEstimateProbability;
            const int slowMs = (i == 0) ? settings_.slowEstimateMs : 0;
            futures.push_back(launch([&graph = graph_, view = views[i], restaurantNode, delivery = order.deliveryNode,
                                      readyAt, now, speed = metersPerSimMs_, inject, slowMs]() {
                if (slowMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slowMs));
                if (inject) throw std::runtime_error("injected estimate failure");
                return estimateDeliveryMs(graph, view, restaurantNode, delivery, readyAt, now, speed);
            }));
        }
        const auto deadline = started + std::chrono::milliseconds(cfg_.dispatch.quoteTimeoutMs);
        for (std::size_t i = 0; i < futures.size(); ++i) {
            estimates_.fetch_add(1, std::memory_order_relaxed);
            const std::future_status status = futures[i].wait_until(deadline);
            if (status == std::future_status::timeout ||
                (status == std::future_status::deferred && std::chrono::steady_clock::now() > deadline)) {
                late_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            try {
                ranked.push_back({futures[i].get(), views[i].index});
            } catch (const std::exception& e) {
                if (failed_.fetch_add(1, std::memory_order_relaxed) < 5)
                    std::cerr << "dispatcher: estimate failed for " << order.id << ": " << e.what() << "\n";
            }
        }
    }
    std::sort(ranked.begin(), ranked.end());
    recordRanking(started);
    return ranked;
}

std::vector<std::pair<long long, int>> Dispatcher::rankOnGpu(const Order& order) {
    const auto started = std::chrono::steady_clock::now();
    const int restaurantNode = restaurants_[static_cast<std::size_t>(order.restaurant)]->node();
    const long long now = clock_.nowMs();
    eta::Couriers couriers;
    std::vector<int> ids;
    for (const std::shared_ptr<Courier>& courier : fleet_.couriers()) {
        const Courier::View v = courier->view();
        if (!v.available) continue;
        try {
            const auto [freeAt, freeNode] = freeAfterBag(graph_, v, now, metersPerSimMs_);
            couriers.freeNode.push_back(freeNode);
            couriers.freeAtMs.push_back(freeAt);
            ids.push_back(v.index);
        } catch (const std::exception&) {
            failed_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::vector<std::pair<long long, int>> ranked;
    if (ids.empty()) return ranked;
    estimates_.fetch_add(static_cast<long long>(ids.size()), std::memory_order_relaxed);
    const eta::Query query{restaurantNode, order.deliveryNode, order.createdAtMs + order.prepMs};
    const eta::Params params{graph_.nodeCount(), now, metersPerHour_, Courier::kHandoverMs};
    std::future<argmin::Key> best = accel_.bestCourier(query, params, std::move(couriers), std::move(ids));
    if (best.wait_until(started + std::chrono::milliseconds(cfg_.dispatch.quoteTimeoutMs)) ==
        std::future_status::timeout) {
        late_.fetch_add(1, std::memory_order_relaxed);
        return ranked;
    }
    try {
        const argmin::Key key = best.get();
        if (key != argmin::kNone && argmin::etaOf(key) != eta::kNoEta)
            ranked.push_back({argmin::etaOf(key), argmin::idOf(key)});
    } catch (const std::exception& e) {
        if (failed_.fetch_add(1, std::memory_order_relaxed) < 5)
            std::cerr << "dispatcher: GPU ranking failed for " << order.id << ": " << e.what() << "\n";
    }
    recordRanking(started);
    return ranked;
}

void Dispatcher::recordRanking(std::chrono::steady_clock::time_point started) {
    const long long us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    rankings_.fetch_add(1, std::memory_order_relaxed);
    totalRankUs_.fetch_add(us, std::memory_order_relaxed);
    if (us > maxRankUs_.load(std::memory_order_relaxed)) maxRankUs_.store(us, std::memory_order_relaxed);
}

std::string Dispatcher::statsLine() const {
    const long long n = rankings_.load(std::memory_order_relaxed);
    std::ostringstream out;
    out << "dispatcher: mode=" << (accel_.gpu() ? "gpu" : settings_.estimatesName()) << " K=" << settings_.candidates
        << " rankings=" << n << " estimates=" << estimates_.load(std::memory_order_relaxed)
        << " late=" << late() << " failed=" << failed()
        << " avgRankUs=" << (n > 0 ? totalRankUs_.load(std::memory_order_relaxed) / n : 0)
        << " maxRankUs=" << maxRankUs_.load(std::memory_order_relaxed);
    return out.str();
}
