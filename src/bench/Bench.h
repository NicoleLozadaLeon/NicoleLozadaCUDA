#pragma once
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "Config.h"
#include "Gpu.h"

namespace bench {
struct Options {
    std::string csv;
    std::string machine = "laptop";
    std::vector<int> threads;
    std::vector<std::string> apps;
    int reps = 10;
    int sizeIndex = -1;
    bool once = false;
};

struct Result {
    double ms = 0.0;
    double kernelMs = -1.0;
    bool correct = false;
};

struct Variant {
    std::string version;
    int threads;
    std::function<Result()> run;
};

class Recorder {
public:
    Recorder(const std::string& path, const std::string& machine);
    void row(const std::string& app, const std::string& folder, const std::string& version,
             const std::string& label, double size, int threads, int rep, double ms, bool correct);

private:
    std::ofstream out_;
    std::string machine_;
};

struct Context {
    const Config& cfg;
    const Options& opt;
    Recorder& rec;
    bool gpu;
};

using AppFn = void (*)(Context&);

struct Registrar {
    Registrar(int order, const char* name, AppFn fn);
};

template <class Fn>
double timeMs(Fn&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

template <class One, class Multi, class Check>
void addCpu(std::vector<Variant>& variants, const Options& opt, One one, Multi multi, Check check) {
    variants.push_back({"cpu1", 1, [=]() {
                            decltype(one()) out;
                            Result r;
                            r.ms = timeMs([&] { out = one(); });
                            r.correct = check(out);
                            return r;
                        }});
    for (int t : opt.threads)
        variants.push_back({"cpuN", t, [=]() {
                                decltype(one()) out;
                                Result r;
                                r.ms = timeMs([&] { out = multi(t); });
                                r.correct = check(out);
                                return r;
                            }});
}

template <class Out, class Gpu, class Check>
void addGpu(std::vector<Variant>& variants, const Context& ctx, const std::string& version, Gpu gpu, Check check) {
    if (!ctx.gpu) return;
    variants.push_back({version, 0, [=]() {
                            Out out;
                            const GpuTiming t = gpu(out);
                            Result r;
                            r.ms = t.e2eMs;
                            r.kernelMs = t.kernelMs;
                            r.correct = check(out);
                            return r;
                        }});
}

bool sizeWanted(const Options& opt, std::size_t index, std::size_t count);
void runCase(Context& ctx, const std::string& app, const std::string& folder, const std::string& label,
             double size, std::vector<Variant>& variants);
int run(const Config& cfg, const Options& opt);
}