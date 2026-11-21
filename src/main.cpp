#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include "Bench.h"
#include "Config.h"
#include "Settings.h"
#include "Simulation.h"
#include "Window.h"
namespace {
std::atomic<bool> g_stopRequested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the stop flag must be lock-free");
void onSignal(int) { g_stopRequested.store(true); }
int usage() {
    std::cerr << "usage: delivery_sim <config.json> [--log <path>] [--headless]\n"
                 "       delivery_sim <config.json> --bench <results.csv> [--bench-once] [--machine <name>]\n"
                 "                    [--threads <n,n,...>] [--reps <n>] [--apps <a,b,...>] [--size <index>]\n";
    return 1;
}
std::vector<std::string> splitList(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');)
        if (!item.empty()) out.push_back(item);
    return out;
}
struct Args {
    std::string config;
    std::string logPath = "events.log";
    bool headless = false;
    bool bench = false;
    bench::Options benchOptions;
};
bool parse(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.config = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&](std::string& out) {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };
        std::string v;
        if (flag == "--log") {
            if (!value(a.logPath)) return false;
        } else if (flag == "--headless") {
            a.headless = true;
        } else if (flag == "--bench") {
            a.bench = true;
            if (!value(a.benchOptions.csv)) return false;
        } else if (flag == "--bench-once") {
            a.bench = true;
            a.benchOptions.once = true;
        } else if (flag == "--machine") {
            if (!value(a.benchOptions.machine)) return false;
        } else if (flag == "--threads") {
            if (!value(v)) return false;
            for (const std::string& t : splitList(v)) a.benchOptions.threads.push_back(std::max(1, std::stoi(t)));
        } else if (flag == "--reps") {
            if (!value(v)) return false;
            a.benchOptions.reps = std::max(1, std::stoi(v));
        } else if (flag == "--apps") {
            if (!value(v)) return false;
            a.benchOptions.apps = splitList(v);
        } else if (flag == "--size") {
            if (!value(v)) return false;
            a.benchOptions.sizeIndex = std::stoi(v);
        } else {
            return false;
        }
    }
    if (a.bench && a.benchOptions.csv.empty()) a.benchOptions.csv = "bench_once.csv";
    return true;
}
void waitHeadless(const Simulation& sim, const Config& cfg, const sigset_t& signals) {
    const long long endMs = static_cast<long long>(cfg.simulation.durationS) * 1000;
    while (!sim.finished()) {
        siginfo_t info;
        if (cfg.simulation.durationS <= 0) {
            if (sigwaitinfo(&signals, &info) > 0) return;
            continue;
        }
        const auto left = sim.clock().toReal(endMs) - std::chrono::steady_clock::now();
        const long long ns = std::max<long long>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(left).count());
        const timespec timeout{static_cast<time_t>(ns / 1000000000LL), static_cast<long>(ns % 1000000000LL)};
        if (sigtimedwait(&signals, &info, &timeout) > 0) return;
    }
}
}
int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse(argc, argv, args)) return usage();
    } catch (const std::exception&) {
        return usage();
    }
    Config cfg;
    try {
        cfg = load_config(args.config);
    } catch (const ConfigError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (args.bench) {
        try {
            return bench::run(cfg, args.benchOptions);
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    if (args.headless) {
        pthread_sigmask(SIG_BLOCK, &signals, nullptr);
    } else {
        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);
    }
    const Settings settings = Settings::fromEnvironment();
    try {
        std::unique_ptr<Window> window;
        if (!args.headless) window = std::make_unique<Window>(cfg);
        Simulation sim(cfg, settings, args.logPath);
        std::cerr << sim.graphReport() << "\n";
        sim.start();
        std::cout << "Simulation has started..." << std::endl;
        if (args.headless) {
            waitHeadless(sim, cfg, signals);
        } else {
            while (!g_stopRequested.load() && !sim.finished()) {
                if (!window->frame(sim)) break;
            }
        }
        sim.stop();
        std::cerr << sim.dispatcher().statsLine() << "\n";
        if (window) std::cerr << "window: frames=" << window->frames() << " fps=" << window->fps() << "\n";
        std::cout << sim.statsJson(args.logPath) << std::endl;
        std::cout << sim.log().summaryJson() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
