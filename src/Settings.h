#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>
// Knobs for the experiments of DP.10. They come from environment variables so that the
// command line stays exactly as the brief defines it. main() reads them ONCE, before any
// thread exists; afterwards the object is only read, so sharing it by const reference
// cannot race. With no variable set, every knob has the value of the final design.
struct Settings {
    enum class Estimates { Pool, Async, Deferred };
    Estimates estimates = Estimates::Pool;   // EQ_ESTIMATES = pool | async | deferred
    int candidates = 8;                      // EQ_CANDIDATES: couriers compared for one order (K)
    bool holdSlotWhileCooking = false;       // EQ_FOOD_WAIT = holding | before
    int slowEstimateMs = 0;                  // EQ_SLOW_ESTIMATE_MS: first estimate of each order is this slow (real ms)
    double failEstimateProbability = 0.0;    // EQ_FAIL_ESTIMATE: chance that one estimate throws
    int graphTasks = 0;                      // EQ_GRAPH_TASKS: 0 = hardware_concurrency()
    int graphGrain = 4;                      // EQ_GRAPH_GRAIN: sources claimed at a time, 0 = static chunks
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