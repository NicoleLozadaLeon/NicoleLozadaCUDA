#include "Simulation.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <sstream>
#include <utility>
#include <nlohmann/json.hpp>
#include "Hotspots.h"
#include "Parallel.h"
namespace {
std::vector<std::shared_ptr<Restaurant>> makeRestaurants(const Config& cfg, const Graph& graph) {
    std::vector<std::shared_ptr<Restaurant>> restaurants;
    for (const RestaurantDef& r : cfg.restaurants)
        restaurants.push_back(std::make_shared<Restaurant>(r.id, r.name, graph.indexOf(r.node), r.pickupSlots));
    return restaurants;
}
std::size_t poolSize(const Settings& settings) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;
    return std::max<std::size_t>(hw, static_cast<std::size_t>(settings.candidates));
}
}
Simulation::Simulation(const Config& cfg, const Settings& settings, const std::string& logPath)
    : cfg_(cfg),
      accel_(settings.gpu, hardwareThreads()),
      graph_(cfg, settings.graphTasks, settings.graphGrain, &accel_),
      clock_(cfg.simulation.timeScale),
      log_(logPath, clock_),
      book_(static_cast<std::size_t>(cfg.orders.maxPending)),
      restaurants_(makeRestaurants(cfg, graph_)),
      fleet_(cfg, settings, graph_, clock_, log_, restaurants_),
      pool_(poolSize(settings)),
      dispatcher_(cfg, settings, graph_, clock_, log_, book_, fleet_, pool_, accel_, restaurants_),
      generator_(cfg, graph_, clock_, log_, book_, restaurants_) {
    if (accel_.gpu()) accel_.keepTable(graph_.distDm(), graph_.nodeCount());
}
Simulation::~Simulation() { stop(); }
void Simulation::start() {
    started_ = true;
    std::vector<std::future<void>> ready;
    for (const std::shared_ptr<Courier>& courier : fleet_.couriers()) {
        std::promise<void> promise;
        ready.push_back(promise.get_future());
        courierThreads_.emplace_back(&Courier::run, courier, std::move(promise));
    }
    {
        std::promise<void> promise;
        ready.push_back(promise.get_future());
        dispatcherThread_ = std::thread(&Dispatcher::run, std::ref(dispatcher_), std::move(promise));
    }
    {
        std::promise<void> promise;
        ready.push_back(promise.get_future());
        generatorThread_ = std::thread(&OrderGenerator::run, std::ref(generator_), std::move(promise));
    }
    for (std::future<void>& f : ready) f.wait();
}
void Simulation::stop() {
    if (stopped_) return;
    stopped_ = true;
    generator_.stop();
    if (generatorThread_.joinable()) generatorThread_.join();
    book_.close();
    fleet_.stop();
    for (const std::shared_ptr<Restaurant>& r : restaurants_) r->stop();
    if (dispatcherThread_.joinable()) dispatcherThread_.join();
    for (std::thread& t : courierThreads_)
        if (t.joinable()) t.join();
    pool_.shutdown();
    if (started_) log_.simulationStopping();
}
bool Simulation::finished() const {
    return cfg_.simulation.durationS > 0 &&
           clock_.nowMs() >= static_cast<long long>(cfg_.simulation.durationS) * 1000;
}
std::string Simulation::graphReport() const {
    std::ostringstream out;
    out << "graph: " << graph_.nodeCount() << " nodes, shortest paths in " << graph_.buildMs() << " ms with "
        << graph_.tasksUsed() << " tasks (grain " << graph_.grain() << ")"
        << (graph_.builtOnGpu() ? ", Floyd-Warshall on the GPU" : "") << ", backend " << accel_.backend();
    return out.str();
}
namespace {
// The second application of each concept (report sections 3.8 to 3.12). Everything comes from what the program
// already has: the graph, Courier::view(), the pending orders of the OrderBook and the events.log.
void secondApplications(const Simulation& sim, const Config& cfg, const std::string& logPath,
                        nlohmann::ordered_json& j) {
    std::vector<Courier::View> views;
    for (const std::shared_ptr<Courier>& c : sim.fleet().couriers()) views.push_back(c->view());
    const long long nowMs = sim.clock().nowMs();

    // 3.8 (03 tiling): street corners with no courier within 500 m. Positions in whole metres (1 m cells).
    const MapGrid metres(cfg.map.bounds, 1.0);
    nearby::Points corners, couriers;
    auto addPoint = [&metres](nearby::Points& p, double lat, double lon) {
        int x = 0, y = 0;
        metres.cellOf(lat, lon, x, y);
        p.x.push_back(x);
        p.y.push_back(y);
    };
    for (int n = 0; n < sim.graph().nodeCount(); ++n) addPoint(corners, sim.graph().lat(n), sim.graph().lon(n));
    for (const Courier::View& v : views) addPoint(couriers, sim.graph().lat(v.legTo), sim.graph().lon(v.legTo));
    const std::vector<int> near = sim.accel().couriersNear(corners, couriers, nearby::kRadiusM);
    int alone = 0;
    nlohmann::ordered_json aloneIds = nlohmann::ordered_json::array();
    for (int n = 0; n < sim.graph().nodeCount(); ++n) {
        if (near[static_cast<std::size_t>(n)] != 0) continue;
        if (++alone <= 3) aloneIds.push_back(sim.graph().idOf(n));
    }
    j["cornersWithoutCourier"] = {{"radiusM", nearby::kRadiusM}, {"corners", alone}, {"first", aloneIds}};

    // 3.9 (04 performance): food that is ready and still waiting in a restaurant for its courier.
    foodwait::Items items;
    for (const Courier::View& v : views)
        for (const Courier::View::Item& it : v.bag) {
            items.readyAtMs.push_back(it.readyAtMs);
            items.pickedUp.push_back(it.pickedUp ? 1 : 0);
        }
    const std::vector<int> wait = sim.accel().foodWaiting(items, nowMs);
    long long waiting = 0, totalMs = 0, maxMs = 0;
    for (int w : wait) {
        if (w <= 0) continue;
        ++waiting;
        totalMs += w;
        maxMs = std::max<long long>(maxMs, w);
    }
    j["foodWaiting"] = {{"ordersInBags", items.size()}, {"waiting", waiting},
                        {"avgMs", waiting ? totalMs / waiting : 0}, {"maxMs", maxMs}};

    // 3.10 (08 reduction): the state of the fleet in one pass. After stop() every view says available = false
    // (the couriers are stopping), so available is taken as "not broken and room in the bag".
    fleetsum::FleetData fleet;
    const std::size_t capacity = static_cast<std::size_t>(cfg.fleet.bagCapacity);
    for (const Courier::View& v : views) {
        unsigned char picked = 0;
        for (const Courier::View::Item& it : v.bag) picked = static_cast<unsigned char>(picked + (it.pickedUp ? 1 : 0));
        fleet.state.push_back(static_cast<unsigned char>(v.state));
        fleet.items.push_back(static_cast<unsigned char>(v.bag.size()));
        fleet.picked.push_back(picked);
        fleet.available.push_back(v.state != Courier::State::Broken && v.bag.size() < capacity ? 1 : 0);
        fleet.legEndMs.push_back(v.legEndMs);
    }
    const fleetsum::Summary s = sim.accel().fleetSummary(fleet);
    auto courierId = [&sim](unsigned idx) -> nlohmann::ordered_json {
        if (idx == fleetsum::kNoIndex || idx >= sim.fleet().couriers().size()) return nullptr;
        return sim.fleet().couriers()[idx]->id();
    };
    j["fleet"] = {{"idle", s.state[fleetsum::kIdle]},
                  {"toRestaurant", s.state[fleetsum::kToRestaurant]},
                  {"carrying", s.state[fleetsum::kCarrying]},
                  {"broken", s.state[fleetsum::kBroken]},
                  {"available", s.available},
                  {"ordersInBags", s.items},
                  {"pickedUp", s.picked},
                  {"fullestBag", courierId(s.mostItemsIdx)},
                  {"latestLegEnd", courierId(s.latestEndIdx)}};

    // 3.11 (09 scan): the queue of orders waiting for a courier, from the events.log this run wrote.
    try {
        const backlog::EventStream events = backlog::loadLog(logPath);
        const backlog::Result r = sim.accel().orderQueue(events);
        const backlog::Summary q = backlog::summarize(events, r);
        j["orderQueue"] = {{"peak", q.peak}, {"peakAtMs", q.peakAtMs}, {"waitingAtEnd", q.waitingAtEnd}};
    } catch (const std::exception& e) {
        j["orderQueue"] = {{"error", e.what()}};
    }

    // 3.12 (05 conv): zones with the most pending orders (25 m cells, as the window) and their nearest courier.
    constexpr double kCellM = 25.0;
    const MapGrid grid(cfg.map.bounds, kCellM);
    std::vector<int> cells;
    for (int node : sim.book().deliveryNodes()) {
        int x = 0, y = 0;
        if (grid.cellOf(sim.graph().lat(node), sim.graph().lon(node), x, y)) cells.push_back(y * grid.width() + x);
    }
    std::vector<hotspots::Cell> courierCells;
    std::vector<std::size_t> courierIndex;
    for (std::size_t i = 0; i < views.size(); ++i) {
        int x = 0, y = 0;
        if (!grid.cellOf(sim.graph().lat(views[i].legTo), sim.graph().lon(views[i].legTo), x, y)) continue;
        courierCells.push_back({x, y});
        courierIndex.push_back(i);
    }
    const Grid demand = sim.accel().pendingDemand(grid.width(), grid.height(), cells);
    nlohmann::ordered_json zones = nlohmann::ordered_json::array();
    for (const hotspots::Hotspot& z : hotspots::rank(demand, kCellM, courierCells)) {
        nlohmann::ordered_json nearest = nullptr;
        if (z.courier >= 0) nearest = sim.fleet().couriers()[courierIndex[static_cast<std::size_t>(z.courier)]]->id();
        zones.push_back({{"cell", {z.x, z.y}},
                         {"pendingPerCell", z.value},
                         {"nearestCourier", nearest},
                         {"distanceM", static_cast<long long>(z.distanceM + 0.5)},
                         {"uncovered", z.uncovered}});
    }
    j["pendingHotspots"] = {{"pendingOrders", cells.size()}, {"zones", zones}};
}
}

std::string Simulation::statsJson(const std::string& logPath) const {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<EventLog::Delivery> deliveries = log_.deliveries();
    std::vector<int> ms, restaurant;
    for (const EventLog::Delivery& d : deliveries) {
        ms.push_back(static_cast<int>(d.deliveredAtMs - d.createdAtMs));
        restaurant.push_back(d.restaurant);
    }
    const int restaurants = static_cast<int>(restaurants_.size());
    nlohmann::ordered_json j;
    j["backend"] = accel_.backend();
    j["deliveryMinutesHistogram"] = accel_.deliveryHistogram(ms, 60000, 120);
    const scan::Grouping groups = accel_.groupByRestaurant(restaurant, restaurants);
    nlohmann::ordered_json perRestaurant = nlohmann::ordered_json::array();
    for (int r = 0; r < restaurants; ++r) {
        const unsigned begin = groups.offsets[static_cast<std::size_t>(r)];
        const unsigned end = groups.offsets[static_cast<std::size_t>(r) + 1];
        long long total = 0;
        for (unsigned i = begin; i < end; ++i) total += ms[static_cast<std::size_t>(groups.order[i])];
        perRestaurant.push_back({{"restaurant", restaurants_[static_cast<std::size_t>(r)]->id()},
                                 {"delivered", end - begin},
                                 {"avgDeliveryMs", end > begin ? total / (end - begin) : 0}});
    }
    j["perRestaurant"] = perRestaurant;
    const MapGrid map(cfg_.map.bounds, 10.0);
    Grid sources = map.empty();
    for (const std::shared_ptr<Courier>& c : fleet_.couriers()) {
        const Courier::View v = c->view();
        map.add(sources, graph_.lat(v.legTo), graph_.lon(v.legTo));
    }
    const Grid field = accel_.coverage(std::move(sources), 100).get();
    std::vector<std::pair<float, int>> nodes;
    for (int n = 0; n < graph_.nodeCount(); ++n) {
        int x = 0, y = 0;
        if (map.cellOf(graph_.lat(n), graph_.lon(n), x, y)) nodes.push_back({field.at(x, y), n});
    }
    std::sort(nodes.begin(), nodes.end());
    nlohmann::ordered_json waiting = nlohmann::ordered_json::array();
    for (std::size_t i = 0; i < nodes.size() && i < 3; ++i) waiting.push_back(graph_.idOf(nodes[i].second));
    j["leastCoveredNodes"] = waiting;
    secondApplications(*this, cfg_, logPath, j);
    j["statsMs"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return j.dump();
}

