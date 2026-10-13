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
namespace {
// Set by the signal handler, read by the main loop once per frame. Inside a signal handler only
// a LOCK-FREE atomic may be touched, so that is checked at compile time.
std::atomic<bool> g_stopRequested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the stop flag must be lock-free");
void onSignal(int) { g_stopRequested.store(true); }
int usage() {
    std::cerr << "usage: delivery_sim <config.json> [--log <path>]\n";
    return 1;
}
}  // namespace
int main(int argc, char** argv) {
    // First thing: a signal that arrives at any later moment only sets the flag.
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
    const Settings settings = Settings::fromEnvironment();   // read once, before any thread
    try {
        Simulation sim(cfg, settings, logPath);
        std::cerr << sim.graphReport() << "\n";
        sim.start();
        std::cout << "Simulation has started..." << std::endl;
        // TEMPORARY until the window (PR 5): one "frame" every 33 ms. A pause is allowed
        // only to pace the frame rate; the window will replace it with cv::waitKey(33).
        while (!g_stopRequested.load() && !sim.finished())
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        sim.stop();
        std::cerr << sim.dispatcher().statsLine() << "\n";
        std::cout << sim.log().summaryJson() << std::endl;   // last line of stdout
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
