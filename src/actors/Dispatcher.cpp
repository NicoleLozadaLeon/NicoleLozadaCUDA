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
long long estimateDeliveryMs(const Graph& graph, const Courier::View& courier, int restaurantNode,
                             int deliveryNode, long long readyAtMs, long long nowMs, double metersPerSimMs) {
    auto travel = [&](int from, int to) {
        const double meters = graph.distance(from, to);
        if (meters == Graph::kUnreachable)
            throw std::runtime_error("c" + std::to_string(courier.index) + " cannot reach node " + graph.idOf(to));
        return static_cast<long long>(meters / metersPerSimMs);
    };
    // Replay the courier's plan: first the orders already in its bag, then the new one.
    long long t = std::max(nowMs, courier.legEndMs);
    int at = courier.legTo;
    for (const Courier::View::Item& item : courier.bag) {
        if (item.pickedUp) continue;
        t = std::max(t + travel(at, item.restaurantNode), item.readyAtMs) + Courier::kHandoverMs;
        at = item.restaurantNode;
    }
    for (const Courier::View::Item& item : courier.bag) {
        t += travel(at, item.deliveryNode);
        at = item.deliveryNode;
    }
    t = std::max(t + travel(at, restaurantNode), readyAtMs) + Courier::kHandoverMs;
    t += travel(restaurantNode, deliveryNode);
    return t - nowMs;
}
Dispatcher::Dispatcher(const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
                       EventLog& log, OrderBook& book, Fleet& fleet, EstimatePool& pool,
                       const std::vector<std::shared_ptr<Restaurant>>& restaurants)
    : cfg_(cfg),
      settings_(settings),
      graph_(graph),
      clock_(clock),
      log_(log),
      book_(book),
      fleet_(fleet),
      pool_(pool),
      restaurants_(restaurants),
      metersPerSimMs_(cfg.fleet.speedKmh / 3600.0),
      rng_(static_cast<std::uint64_t>(cfg.orders.seed) + 7u) {}
void Dispatcher::run(std::promise<void> ready) {
    ready.set_value();
    // take() blocks without CPU and returns std::nullopt once the book is closed.
    while (std::optional<Order> order = book_.take()) {
        if (!dispatch(std::move(*order))) break;
    }
}
bool Dispatcher::dispatch(Order order) {
    const long long acceptDeadline = order.createdAtMs + cfg_.dispatch.acceptTimeoutMs;
    while (true) {
        if (fleet_.stopping()) return false;   // the order ends as pending
        const unsigned long long seen = fleet_.version();
        for (const std::pair<long long, int>& candidate : rankCouriers(order)) {
            // The courier re-checks its state under its own lock; on success `order` is moved.
            if (fleet_.couriers()[static_cast<std::size_t>(candidate.second)]->offer(order)) return true;
        }
        if (clock_.nowMs() >= acceptDeadline) {
            log_.orderRejected(order.id, "noCourier");
            return true;
        }
        // Nobody can take it now: sleep until a bag gets room, a courier is repaired,
        // the accept deadline passes or the simulation stops. No polling.
        if (!fleet_.waitForChange(seen, clock_.toReal(acceptDeadline))) return false;

    }
}
std::future<long long> Dispatcher::launch(std::function<long long()> estimate) {
    switch (settings_.estimates) {
        case Settings::Estimates::Async:
            return std::async(std::launch::async, std::move(estimate));     // one new thread per estimate
        case Settings::Estimates::Deferred:
            return std::async(std::launch::deferred, std::move(estimate));  // runs later, inside get()
        case Settings::Estimates::Pool:
            break;
    }
    return pool_.submit(std::move(estimate));   // final design
}
std::vector<std::pair<long long, int>> Dispatcher::rankCouriers(const Order& order) {
    const auto started = std::chrono::steady_clock::now();
    const int restaurantNode = restaurants_[static_cast<std::size_t>(order.restaurant)]->node();
    const long long readyAt = order.createdAtMs + order.prepMs;
    const long long now = clock_.nowMs();
    // 1. Copies of the available couriers, taken one lock at a time (never two at once).
    std::vector<Courier::View> views;
    for (const std::shared_ptr<Courier>& courier : fleet_.couriers()) {
        Courier::View v = courier->view();
        if (v.available) views.push_back(std::move(v));
    }
    // 2. Keep the K closest to the restaurant: a cheap table lookup each.
    const std::size_t k = std::min(views.size(), static_cast<std::size_t>(settings_.candidates));
    std::partial_sort(views.begin(), views.begin() + static_cast<std::ptrdiff_t>(k), views.end(),
                      [&](const Courier::View& a, const Courier::View& b) {
                          return graph_.distance(a.legTo, restaurantNode) < graph_.distance(b.legTo, restaurantNode);
                      });
    views.resize(k);
    std::vector<std::pair<long long, int>> ranked;
    {
        // 3. Fan-out: every estimate starts now. Each task gets COPIES of what it needs
        //    (the view, the numbers) and a reference only to the const Graph.
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::vector<std::future<long long>> futures;
        futures.reserve(views.size());
        for (std::size_t i = 0; i < views.size(); ++i) {
            const bool inject = unit(rng_) < settings_.failEstimateProbability;
            const int slowMs = (i == 0) ? settings_.slowEstimateMs : 0;
            futures.push_back(launch([&graph = graph_, view = views[i], restaurantNode, delivery = order.deliveryNode,
                                      readyAt, now, speed = metersPerSimMs_, inject, slowMs]() {
                if (slowMs > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slowMs));   // experiment only
                if (inject) throw std::runtime_error("injected estimate failure");               // experiment only
                return estimateDeliveryMs(graph, view, restaurantNode, delivery, readyAt, now, speed);
            }));
        }
        // 4. Fan-in with ONE deadline for all of them. Late estimates are ignored; a failed one
        //    arrives here as an exception from get() and is counted, never fatal.
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
                ranked.push_back({futures[i].get(), views[i].index});   // a deferred task runs here
            } catch (const std::exception& e) {
                if (failed_.fetch_add(1, std::memory_order_relaxed) < 5)
                    std::cerr << "dispatcher: estimate failed for " << order.id << ": " << e.what() << "\n";
            }
        }
    }   // `futures` dies here. A std::async future of a late task BLOCKS in its destructor;
        // a pool (packaged_task) future does not. Experiment E1 measures exactly this.
    std::sort(ranked.begin(), ranked.end());
    const long long us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    rankings_.fetch_add(1, std::memory_order_relaxed);
    totalRankUs_.fetch_add(us, std::memory_order_relaxed);
    if (us > maxRankUs_.load(std::memory_order_relaxed)) maxRankUs_.store(us, std::memory_order_relaxed);   // one writer
    return ranked;
}

std::string Dispatcher::statsLine() const {
    const long long n = rankings_.load(std::memory_order_relaxed);
    std::ostringstream out;
    out << "dispatcher: mode=" << settings_.estimatesName() << " K=" << settings_.candidates
        << " rankings=" << n << " estimates=" << estimates_.load(std::memory_order_relaxed)
        << " late=" << late() << " failed=" << failed()
        << " avgRankUs=" << (n > 0 ? totalRankUs_.load(std::memory_order_relaxed) / n : 0)
        << " maxRankUs=" << maxRankUs_.load(std::memory_order_relaxed);
    return out.str();
}