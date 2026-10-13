#include "Courier.h"
#include <algorithm>
#include <cstdint>
#include <utility>
#include "Fleet.h"
Courier::Courier(int index, const Config& cfg, const Settings& settings, const Graph& graph, const SimClock& clock,
                 EventLog& log, const std::vector<std::shared_ptr<Restaurant>>& restaurants, Fleet& fleet)
    : index_(index),
      id_("c" + std::to_string(index)),
      capacity_(static_cast<std::size_t>(cfg.fleet.bagCapacity)),
      metersPerSimMs_(cfg.fleet.speedKmh / 3600.0),   // km/h -> m per ms
      breakdownProbability_(cfg.incidents.breakdownProbability),
      settings_(settings),
      graph_(graph),
      clock_(clock),
      log_(log),
      restaurants_(restaurants),
      fleet_(fleet),
      rng_(static_cast<std::uint64_t>(cfg.orders.seed) + 1000u + static_cast<std::uint64_t>(index)),
      node_(graph.indexOf(cfg.fleet.startNode)),
      legFrom_(node_),
      legTo_(node_) {}
void Courier::run(std::promise<void> ready) {
    ready.set_value();   // one-shot "I am ready" for the start line
    Task task{};
    while (nextTask(task)) {
        const bool goOn = task.pickup ? pickupAt(task.restaurant) : deliverAt(task.node);
        if (!goOn) break;
    }
}

void Courier::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
}
bool Courier::nextTask(Task& task) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (bag_.empty()) state_ = State::Idle;
    // Nothing to do: sleep without using CPU until an order arrives or the simulation stops.
    wake_.wait(lock, [this] { return stopping_ || !bag_.empty(); });
    if (stopping_) return false;
    for (const BagItem& item : bag_) {
        if (!item.pickedUp) {   // collect first, in the order the orders were assigned
            task = {true, item.order.restaurant, item.restaurantNode};
            return true;
        }
    }
    int best = bag_.front().order.deliveryNode;   // everything collected: nearest customer first
    for (const BagItem& item : bag_)
        if (graph_.distance(node_, item.order.deliveryNode) < graph_.distance(node_, best))
            best = item.order.deliveryNode;
    task = {false, -1, best};
    return true;
}
bool Courier::pickupAt(int r) {
    Restaurant& restaurant = *restaurants_[static_cast<std::size_t>(r)];
    const Ride ride = rideTo(restaurant.node(), State::ToRestaurant);
    if (ride == Ride::Stopped) return false;
    if (ride == Ride::Stuck) return waitForStop();
    if (ride == Ride::BrokeDown) return true;   // the bag may have changed: plan again
    long long readyAt = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const BagItem& item : bag_)
            if (!item.pickedUp && item.order.restaurant == r) readyAt = std::max(readyAt, item.readyAtMs);
    }
    // Final design: wait for the food BEFORE taking a slot, so the slot stays free for others.
    if (!settings_.holdSlotWhileCooking && !sleepUntil(readyAt)) return false;
    if (!restaurant.acquireSlot()) return false;   // FIFO turn; blocks without CPU
    SlotGuard slot(restaurant);                    // released on every way out of this function
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (BagItem& item : bag_) {
            if (!item.pickedUp && item.order.restaurant == r) {
                item.pickupStarted = true;
                readyAt = std::max(readyAt, item.readyAtMs);
                log_.pickupStarted(item.order.id, id_, restaurant.id());
            }
        }
    }
    if (settings_.holdSlotWhileCooking && !sleepUntil(readyAt)) return false;   // experiment only
    if (!sleepUntil(clock_.nowMs() + kHandoverMs)) return false;               // food changes hands
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (BagItem& item : bag_) {
            if (item.pickupStarted && !item.pickedUp) {
                item.pickedUp = true;
                log_.orderPickedUp(item.order.id, id_, restaurant.id());
            }
        }
    }
    return true;
}
bool Courier::deliverAt(int node) {
    const Ride ride = rideTo(node, State::Carrying);
    if (ride == Ride::Stopped) return false;
    if (ride == Ride::Stuck) return waitForStop();
    if (ride == Ride::BrokeDown) return true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = bag_.begin(); it != bag_.end();) {
            if (it->pickedUp && it->order.deliveryNode == node) {
                log_.orderDelivered(it->order.id, id_);
                it = bag_.erase(it);
            } else {
                ++it;
            }
        }
    }
    fleet_.notifyChanged();   // room in the bag again (after releasing our own lock)
    return true;
}
Courier::Ride Courier::rideTo(int target, State state) {
    const std::vector<int> path = graph_.route(node_, target);
    if (path.empty()) return Ride::Stuck;   // no street leads there from here
    const int legs = static_cast<int>(path.size()) - 1;
    int breakAfterLeg = -1;   // a breakdown happens at a random moment of this trip, or never
    if (legs > 0 && std::bernoulli_distribution(breakdownProbability_)(rng_))
        breakAfterLeg = std::uniform_int_distribution<int>(0, legs - 1)(rng_);
    for (int i = 0; i < legs; ++i) {
        const int a = path[static_cast<std::size_t>(i)];
        const int b = path[static_cast<std::size_t>(i) + 1];
        const long long start = clock_.nowMs();
        const long long end = start + legMs(graph_.distance(a, b));
        {
            std::lock_guard<std::mutex> lock(mutex_);   // the window reads these under the same lock
            state_ = state;
            legFrom_ = a;
            legTo_ = b;
            legStartMs_ = start;
            legEndMs_ = end;
        }
        if (!sleepUntil(end)) return Ride::Stopped;   // riding = simulated time passing
        {
            std::lock_guard<std::mutex> lock(mutex_);
            node_ = b;
            legFrom_ = legTo_ = b;
            legStartMs_ = legEndMs_ = end;
        }
        if (i == breakAfterLeg) return breakDown() ? Ride::BrokeDown : Ride::Stopped;
    }
    return Ride::Arrived;
}
bool Courier::breakDown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        broken_ = true;
        state_ = State::Broken;
        log_.courierBreakdown(id_);
    }
    const long long repairedAt = clock_.nowMs() + kRepairMs;
    while (true) {
        const unsigned long long seen = fleet_.version();   // read BEFORE trying: no change is missed
        if (fleet_.handOver(*this)) break;                  // all orders moved (or the bag was empty)
        if (clock_.nowMs() >= repairedAt) break;            // repaired first: keep the orders
        if (!fleet_.waitForChange(seen, clock_.toReal(repairedAt))) return false;   // stopping
    }
    if (!sleepUntil(repairedAt)) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        broken_ = false;
        state_ = bag_.empty() ? State::Idle : State::Carrying;
    }
    fleet_.notifyChanged();   // available again
    return true;
}
bool Courier::waitForStop() {
    // Stuck in a part of the map with no way out: wait (no CPU) for the end of the simulation.
    std::unique_lock<std::mutex> lock(mutex_);
    wake_.wait(lock, [this] { return stopping_; });
    return false;
}
bool Courier::sleepUntil(long long simMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    // Timed wait on our own condition variable: stop() interrupts it at once.
    return !wake_.wait_until(lock, clock_.toReal(simMs), [this] { return stopping_; });
}
long long Courier::legMs(double meters) const {
    return static_cast<long long>(meters / metersPerSimMs_);
}
bool Courier::offer(Order& order) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (broken_ || stopping_ || bag_.size() >= capacity_) return false;   // bag capacity, at every instant
    log_.orderAssigned(order.id, id_);   // logged before the courier can act on it
    const int restaurantNode = restaurants_[static_cast<std::size_t>(order.restaurant)]->node();
    const long long readyAt = order.createdAtMs + order.prepMs;
    bag_.push_back(BagItem{std::move(order), restaurantNode, readyAt});
    wake_.notify_all();
    return true;

}

Courier::View Courier::view() const {
    std::lock_guard<std::mutex> lock(mutex_);
    View v;
    v.index = index_;
    v.state = state_;
    v.available = !broken_ && !stopping_ && bag_.size() < capacity_;
    v.legFrom = legFrom_;
    v.legTo = legTo_;
    v.legStartMs = legStartMs_;
    v.legEndMs = legEndMs_;
    v.bag.reserve(bag_.size());
    for (const BagItem& item : bag_)
        v.bag.push_back({item.restaurantNode, item.order.deliveryNode, item.readyAtMs, item.pickedUp});
    return v;
}
bool Courier::transferAll(Courier& from, Courier& to, EventLog& log) {
    if (&from == &to) return false;
    // Both mutexes at once, with the deadlock-avoidance algorithm: two simultaneous
    // breakdowns can never block each other, whatever order the couriers are passed in.
    std::scoped_lock both(from.mutex_, to.mutex_);
    if (from.bag_.empty()) return true;
    if (to.broken_ || to.stopping_ || to.capacity_ - to.bag_.size() < from.bag_.size()) return false;
    for (BagItem& item : from.bag_) {
        log.orderReassigned(item.order.id, from.id_, to.id_);
        to.bag_.push_back(std::move(item));
    }
    from.bag_.clear();   // nobody can look between the moves: both locks are held
    to.wake_.notify_all();
    return true;
}
