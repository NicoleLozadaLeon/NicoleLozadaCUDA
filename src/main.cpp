#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include "Config.h"
#include "Settings.h"
#include "Simulation.h"
#include "Window.h"
namespace {
std::atomic<bool> g_stopRequested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the stop flag must be lock-free");
void onSignal(int) { g_stopRequested.store(true); }
int usage() {
    std::cerr << "usage: delivery_sim <config.json> [--log <path>]\n";
    return 1;
}
}
int main(int argc, char** argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    if (argc != 2 && argc != 4) return usage();
    std::string logPath = "events.log";
    if (argc == 4) {
        if (std::string(argv[2]) != "--log") return usage();
        logPath = argv[3];
    }
    Config cfg;
    try {
        cfg = load_config(argv[1]);
    } catch (const ConfigError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    const Settings settings = Settings::fromEnvironment();
    try {
        Window window(cfg);
        Simulation sim(cfg, settings, logPath);
        std::cerr << sim.graphReport() << "\n";
        sim.start();
        std::cout << "Simulation has started..." << std::endl;
        while (!g_stopRequested.load() && !sim.finished()) {
            if (!window.frame(sim)) break;
        }
        sim.stop();
        std::cerr << sim.dispatcher().statsLine() << "\n";
        std::cout << sim.log().summaryJson() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
