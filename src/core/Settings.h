#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>
struct Settings {
    enum class Estimates { Pool, Async, Deferred };
    Estimates estimates = Estimates::Pool;
    int candidates = 8;
    bool holdSlotWhileCooking = false;
    int slowEstimateMs = 0;
    double failEstimateProbability = 0.0;
    int graphTasks = 0;
    int graphGrain = 4;
    const char* estimatesName() const {
        switch (estimates) {
            case Estimates::Async: return "async";
            case Estimates::Deferred: return "deferred";
            case Estimates::Pool: break;
        }
        return "pool";
    }
    static Settings fromEnvironment() {
        Settings s;
        if (const char* v = std::getenv("EQ_ESTIMATES")) {
            const std::string mode(v);
            if (mode == "async") s.estimates = Estimates::Async;
            else if (mode == "deferred") s.estimates = Estimates::Deferred;
        }
        if (const char* v = std::getenv("EQ_CANDIDATES")) s.candidates = std::max(1, std::atoi(v));
        if (const char* v = std::getenv("EQ_FOOD_WAIT")) s.holdSlotWhileCooking = std::string(v) == "holding";
        if (const char* v = std::getenv("EQ_SLOW_ESTIMATE_MS")) s.slowEstimateMs = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("EQ_FAIL_ESTIMATE")) s.failEstimateProbability = std::clamp(std::atof(v), 0.0, 1.0);
        if (const char* v = std::getenv("EQ_GRAPH_TASKS")) s.graphTasks = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("EQ_GRAPH_GRAIN")) s.graphGrain = std::max(0, std::atoi(v));
        return s;
    }
};
