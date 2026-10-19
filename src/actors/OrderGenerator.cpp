#include "OrderGenerator.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
OrderGenerator::OrderGenerator(const Config& cfg, const Graph& graph, const SimClock& clock, EventLog& log,
                               OrderBook& book, const std::vector<std::shared_ptr<Restaurant>>& restaurants)
    : cfg_(cfg),
      graph_(graph),
      clock_(clock),
      log_(log),
      book_(book),
      restaurants_(restaurants),
      rng_(static_cast<std::uint64_t>(cfg.orders.seed)) {}
void OrderGenerator::run(std::promise<void> ready) {
    ready.set_value();
    // One arrival out of ten is a burst of 2..burstMax orders at the same instant. The gaps are
    // stretched by the mean burst size, so the mean time between two orders is meanIntervalMs.
    const int burstMax = cfg_.orders.burstMax;
    const double meanBurst = burstMax >= 2 ? 0.9 + 0.1 * (2.0 + burstMax) / 2.0 : 1.0;
    const double meanGapMs = std::max(1, cfg_.orders.meanIntervalMs) * meanBurst;
    std::exponential_distribution<double> gap(1.0 / meanGapMs);
    std::bernoulli_distribution isBurst(burstMax >= 2 ? 0.1 : 0.0);
    std::uniform_int_distribution<int> burstSize(2, std::max(2, burstMax));
    double nextMs = 0.0;
    while (true) {
        nextMs += gap(rng_);
        if (!sleepUntil(static_cast<long long>(nextMs))) return;
        const int n = isBurst(rng_) ? burstSize(rng_) : 1;
        for (int i = 0; i < n; ++i) createOrder();
    }
}
void OrderGenerator::createOrder() {
    std::uniform_int_distribution<std::size_t> pickRestaurant(0, restaurants_.size() - 1);
    std::uniform_int_distribution<int> pickNode(0, graph_.nodeCount() - 1);
    const std::size_t r = pickRestaurant(rng_);
    const RestaurantDef& def = cfg_.restaurants[r];
    const int delivery = pickNode(rng_);
    const long long prepMs = std::uniform_int_distribution<long long>(def.prepTimeMs[0], def.prepTimeMs[1])(rng_);
    const std::string id = "o" + std::to_string(nextId_++);
    const long long createdAt = log_.orderCreated(id, def.id);
    if (!graph_.reachable(restaurants_[r]->node(), delivery)) {
        log_.orderRejected(id, "unreachable");
        return;
    }
    Order order(id, static_cast<int>(r), delivery, createdAt, prepMs);
    if (!book_.submit(std::move(order)))   // moved only if accepted
        log_.orderRejected(id, "queueFull");
}
bool OrderGenerator::sleepUntil(long long simMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    return !wake_.wait_until(lock, clock_.toReal(simMs), [this] { return stopping_; });
}
void OrderGenerator::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
}