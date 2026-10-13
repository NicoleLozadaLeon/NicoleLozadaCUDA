#include "Simulation.h"
#include <algorithm>
#include <functional>
#include <future>
#include <sstream>
#include <utility>
namespace {
std::vector<std::shared_ptr<Restaurant>> makeRestaurants(const Config& cfg, const Graph& graph) {
    std::vector<std::shared_ptr<Restaurant>> restaurants;
    for (const RestaurantDef& r : cfg.restaurants)
        restaurants.push_back(std::make_shared<Restaurant>(r.id, r.name, graph.indexOf(r.node), r.pickupSlots));
    return restaurants;
}
// All K estimates of one order must be able to run at the same time, and there is no gain in
// more busy threads than cores: the pool gets max(cores, K) workers.
std::size_t poolSize(const Settings& settings) {
    unsigned hw = std::thread::hardware_concurrency();   // 0 means unknown
    if (hw == 0) hw = 2;
    return std::max<std::size_t>(hw, static_cast<std::size_t>(settings.candidates));
}
}  // namespace
Simulation::Simulation(const Config& cfg, const Settings& settings, const std::string& logPath)
    : cfg_(cfg),
      graph_(cfg, settings.graphTasks, settings.graphGrain),
      clock_(cfg.simulation.timeScale),
      log_(logPath, clock_),
      book_(static_cast<std::size_t>(cfg.orders.maxPending)),
      restaurants_(makeRestaurants(cfg, graph_)),
      fleet_(cfg, settings, graph_, clock_, log_, restaurants_),
      pool_(poolSize(settings)),
      dispatcher_(cfg, settings, graph_, clock_, log_, book_, fleet_, pool_, restaurants_),
      generator_(cfg, graph_, clock_, log_, book_, restaurants_) {}
Simulation::~Simulation() { stop(); }
void Simulation::start() {
    started_ = true;
    std::vector<std::future<void>> ready;
    for (const std::shared_ptr<Courier>& courier : fleet_.couriers()) {
        std::promise<void> promise;
        ready.push_back(promise.get_future());
        // A member function on a thread. The shared_ptr is COPIED into the thread, so the courier
        // lives at least as long as its thread; the promise is MOVED in (it is move-only).
        courierThreads_.emplace_back(&Courier::run, courier, std::move(promise));
    }
    {
        std::promise<void> promise;
        ready.push_back(promise.get_future());
        // BY REFERENCE (std::ref): the dispatcher is a member of this object, and this object
        // joins the thread in stop() before the dispatcher is destroyed.
        dispatcherThread_ = std::thread(&Dispatcher::run, std::ref(dispatcher_), std::move(promise));
    }
    {
        std::promise<void> promise;
        ready.push_back(promise.get_future());
        generatorThread_ = std::thread(&OrderGenerator::run, std::ref(generator_), std::move(promise));
    }
    for (std::future<void>& f : ready) f.wait();   // every thread said "ready": no sleep as a barrier
}
void Simulation::stop() {
    if (stopped_) return;
    stopped_ = true;
    // 1. No more orders.
    generator_.stop();
    if (generatorThread_.joinable()) generatorThread_.join();
    // 2. Wake every thread from whatever it waits on: book, fleet, restaurants, its own clock.
    book_.close();
    fleet_.stop();
    for (const std::shared_ptr<Restaurant>& r : restaurants_) r->stop();
    // 3. Join them all: none abandoned, none killed by the process exiting.
    if (dispatcherThread_.joinable()) dispatcherThread_.join();
    for (std::thread& t : courierThreads_)
        if (t.joinable()) t.join();
    pool_.shutdown();
    // 4. Last line of the log. Orders still in the book or in a bag are pending:
    //    the summary computes pending = created - delivered - rejected.
    if (started_) log_.simulationStopping();
}
bool Simulation::finished() const {
    return cfg_.simulation.durationS > 0 &&
           clock_.nowMs() >= static_cast<long long>(cfg_.simulation.durationS) * 1000;
}
std::string Simulation::graphReport() const {
    std::ostringstream out;
    out << "graph: " << graph_.nodeCount() << " nodes, shortest paths in " << graph_.buildMs() << " ms with "
        << graph_.tasksUsed() << " tasks (grain " << graph_.grain() << ")";
    return out.str();
}