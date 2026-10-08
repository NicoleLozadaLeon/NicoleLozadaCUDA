#pragma once
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Bounds { double north, south, west, east; };
struct MapInfo { std::string image; std::string attribution; Bounds bounds; };
struct NodeDef { std::string id; double lat, lon; };
struct StreetDef { std::string id, from, to; bool oneWay; };
struct RestaurantDef { std::string id, name, node; int pickupSlots; std::array<int, 2> prepTimeMs; };
struct FleetDef { int couriers; int bagCapacity; double speedKmh; std::string startNode; };
struct OrdersDef { int meanIntervalMs; int burstMax; int maxPending; long long seed; };
struct DispatchDef { int quoteTimeoutMs; int acceptTimeoutMs; };
struct IncidentsDef { double breakdownProbability; };
struct SimulationDef { int durationS; double timeScale; };

struct Config {
    MapInfo map;
    std::vector<NodeDef> nodes;
    std::vector<StreetDef> streets;
    std::vector<RestaurantDef> restaurants;
    FleetDef fleet;
    OrdersDef orders;
    DispatchDef dispatch;
    IncidentsDef incidents;
    SimulationDef simulation;
};

// Throws ConfigError with a readable message on any problem.
// The returned map.image path is already resolved relative to the config file.
Config load_config(const std::string& path);
