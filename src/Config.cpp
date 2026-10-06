#include "Config.h"

#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

// Reads j.at(key) as T. A missing key or a wrong type becomes a ConfigError naming the key.
template <typename T>
T require(const json& j, const char* key, const std::string& where) {
    try {
        return j.at(key).get<T>();
    } catch (const json::exception&) {
        throw ConfigError(where + ": missing or invalid property '" + key + "'");
    }
}

// Returns j.at(key) after checking it is an object (or an array).
const json& require_object(const json& j, const char* key, const char* where) {
    if (!j.is_object() || !j.contains(key) || !j.at(key).is_object())
        throw ConfigError(std::string(where) + ": missing or invalid object '" + key + "'");
    return j.at(key);
}

const json& require_array(const json& j, const char* key, const char* where) {
    if (!j.is_object() || !j.contains(key) || !j.at(key).is_array())
        throw ConfigError(std::string(where) + ": missing or invalid array '" + key + "'");
    return j.at(key);
}

// Folder of the config file, with a trailing slash ("" if the path has no folder).
std::string folder_of(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : path.substr(0, pos + 1);
}

}  // namespace

Config load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw ConfigError("cannot read configuration file: " + path);

    json root;
    try {
        in >> root;
    } catch (const json::exception& e) {
        throw ConfigError(std::string("malformed JSON in ") + path + ": " + e.what());
    }
    if (!root.is_object()) throw ConfigError("the configuration must be a JSON object");

    Config cfg;
    const json& m = require_object(root, "map", "config");
    const std::string image = require<std::string>(m, "image", "map");
    cfg.map.image = (!image.empty() && image[0] == '/') ? image : folder_of(path) + image;
    cfg.map.attribution = require<std::string>(m, "attribution", "map");
    const json& b = require_object(m, "bounds", "map");
    cfg.map.bounds = {require<double>(b, "north", "map.bounds"), require<double>(b, "south", "map.bounds"),
                      require<double>(b, "west", "map.bounds"), require<double>(b, "east", "map.bounds")};
    if (cfg.map.bounds.north <= cfg.map.bounds.south || cfg.map.bounds.east <= cfg.map.bounds.west)
        throw ConfigError("map.bounds: north must be greater than south and east greater than west");
    std::set<std::string> node_ids;
    for (const json& n : require_array(root, "nodes", "config")) {
        NodeDef nd;
        nd.id = require<std::string>(n, "id", "node");
        nd.lat = require<double>(n, "lat", "node " + nd.id);
        nd.lon = require<double>(n, "lon", "node " + nd.id);
        if (!node_ids.insert(nd.id).second)
            throw ConfigError("duplicate node id '" + nd.id + "'");
        cfg.nodes.push_back(nd);
    }
    if (cfg.nodes.empty())
        throw ConfigError("config: 'nodes' must contain at least one node");

    for (const json& s : require_array(root, "streets", "config")) {
        StreetDef sd;
        sd.id = require<std::string>(s, "id", "street");
        sd.from = require<std::string>(s, "from", "street " + sd.id);
        sd.to = require<std::string>(s, "to", "street " + sd.id);
        sd.oneWay = require<bool>(s, "oneWay", "street " + sd.id);
        if (node_ids.count(sd.from) == 0)
            throw ConfigError("street '" + sd.id + "' refers to unknown node '" + sd.from + "'");
        if (node_ids.count(sd.to) == 0)
            throw ConfigError("street '" + sd.id + "' refers to unknown node '" + sd.to + "'");
        cfg.streets.push_back(sd);
    }

    for (const json& r : require_array(root, "restaurants", "config")) {
        RestaurantDef rd;
        rd.id = require<std::string>(r, "id", "restaurant");
        rd.name = require<std::string>(r, "name", "restaurant " + rd.id);
        rd.node = require<std::string>(r, "node", "restaurant " + rd.id);
        rd.pickupSlots = require<int>(r, "pickupSlots", "restaurant " + rd.id);
        rd.prepTimeMs = require<std::array<int, 2>>(r, "prepTimeMs", "restaurant " + rd.id);
        if (node_ids.count(rd.node) == 0)
            throw ConfigError("restaurant '" + rd.id + "' refers to unknown node '" + rd.node + "'");
        if (rd.pickupSlots < 1)
            throw ConfigError("restaurant '" + rd.id + "': pickupSlots must be at least 1");
        if (rd.prepTimeMs[0] < 0 || rd.prepTimeMs[0] > rd.prepTimeMs[1])
            throw ConfigError("restaurant '" + rd.id + "': prepTimeMs must be [min, max] with 0 <= min <= max");
        cfg.restaurants.push_back(rd);
    }
    if (cfg.restaurants.empty())
        throw ConfigError("config: 'restaurants' must contain at least one restaurant");
    const json& f = require_object(root, "fleet", "config");
    cfg.fleet.couriers = require<int>(f, "couriers", "fleet");
    cfg.fleet.bagCapacity = require<int>(f, "bagCapacity", "fleet");
    cfg.fleet.speedKmh = require<double>(f, "speedKmh", "fleet");
    cfg.fleet.startNode = require<std::string>(f, "startNode", "fleet");
    if (cfg.fleet.couriers < 1)
        throw ConfigError("fleet.couriers must be at least 1");
    if (cfg.fleet.bagCapacity < 1)
        throw ConfigError("fleet.bagCapacity must be at least 1");
    if (cfg.fleet.speedKmh <= 0)
        throw ConfigError("fleet.speedKmh must be greater than 0");
    if (node_ids.count(cfg.fleet.startNode) == 0)
        throw ConfigError("fleet.startNode refers to unknown node '" + cfg.fleet.startNode + "'");

    const json& o = require_object(root, "orders", "config");
    cfg.orders.meanIntervalMs = require<int>(o, "meanIntervalMs", "orders");
    cfg.orders.burstMax = require<int>(o, "burstMax", "orders");
    cfg.orders.maxPending = require<int>(o, "maxPending", "orders");
    cfg.orders.seed = require<long long>(o, "seed", "orders");
    if (cfg.orders.meanIntervalMs < 1)
        throw ConfigError("orders.meanIntervalMs must be at least 1");
    if (cfg.orders.burstMax < 1)
        throw ConfigError("orders.burstMax must be at least 1");
    if (cfg.orders.maxPending < 0)
        throw ConfigError("orders.maxPending must not be negative");

    const json& d = require_object(root, "dispatch", "config");
    cfg.dispatch.quoteTimeoutMs = require<int>(d, "quoteTimeoutMs", "dispatch");
    cfg.dispatch.acceptTimeoutMs = require<int>(d, "acceptTimeoutMs", "dispatch");
    if (cfg.dispatch.quoteTimeoutMs < 0 || cfg.dispatch.acceptTimeoutMs < 0)
        throw ConfigError("dispatch timeouts must not be negative");

    const json& inc = require_object(root, "incidents", "config");
    cfg.incidents.breakdownProbability = require<double>(inc, "breakdownProbability", "incidents");
    if (cfg.incidents.breakdownProbability < 0 || cfg.incidents.breakdownProbability > 1)
        throw ConfigError("incidents.breakdownProbability must be between 0 and 1");

    const json& sim = require_object(root, "simulation", "config");
    cfg.simulation.durationS = require<int>(sim, "durationS", "simulation");
    cfg.simulation.timeScale = require<double>(sim, "timeScale", "simulation");
    if (cfg.simulation.durationS < 0)
        throw ConfigError("simulation.durationS must not be negative");
    if (cfg.simulation.timeScale <= 0)
        throw ConfigError("simulation.timeScale must be greater than 0");

    return cfg;
}
