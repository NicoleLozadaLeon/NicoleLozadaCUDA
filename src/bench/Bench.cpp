#include "Bench.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include "Parallel.h"

namespace bench {
Recorder::Recorder(const std::string& path, const std::string& machine) : out_(path), machine_(machine) {
    if (!out_) throw std::runtime_error("cannot write the benchmark file: " + path);
    out_ << "machine,app,concept,version,size_label,size,threads,rep,ms,correct\n";
}

void Recorder::row(const std::string& app, const std::string& folder, const std::string& version,
                   const std::string& label, double size, int threads, int rep, double ms, bool correct) {
    out_ << machine_ << ',' << app << ',' << folder << ',' << version << ",\"" << label << "\"," << std::fixed
         << std::setprecision(0) << size << ',' << threads << ',' << rep << ',' << std::setprecision(4) << ms << ','
         << (correct ? "yes" : "no") << '\n';
    out_.flush();
}

bool sizeWanted(const Options& opt, std::size_t index, std::size_t count) {
    if (opt.once) return index == count / 2;
    if (opt.sizeIndex >= 0) return index == static_cast<std::size_t>(opt.sizeIndex);
    return true;
}

void runCase(Context& ctx, const std::string& app, const std::string& folder, const std::string& label,
             double size, std::vector<Variant>& variants) {
    for (Variant& v : variants) v.run();
    std::map<std::string, std::vector<double>> times;
    bool allCorrect = true;
    const int reps = ctx.opt.once ? 1 : ctx.opt.reps;
    for (int rep = 0; rep < reps; ++rep) {
        for (std::size_t i = 0; i < variants.size(); ++i) {
            Variant& v = variants[(i + static_cast<std::size_t>(rep)) % variants.size()];
            const Result r = v.run();
            allCorrect = allCorrect && r.correct;
            const std::string tag = v.version + (v.threads > 1 ? std::to_string(v.threads) : "");
            if (r.kernelMs >= 0.0) {
                ctx.rec.row(app, folder, v.version + "_kernel", label, size, v.threads, rep, r.kernelMs, r.correct);
                ctx.rec.row(app, folder, v.version + "_e2e", label, size, v.threads, rep, r.ms, r.correct);
                times[tag + "_kernel"].push_back(r.kernelMs);
                times[tag + "_e2e"].push_back(r.ms);
            } else {
                ctx.rec.row(app, folder, v.version, label, size, v.threads, rep, r.ms, r.correct);
                times[tag].push_back(r.ms);
            }
        }
    }
    std::cerr << app << " " << label << ":";
    for (auto& [tag, list] : times) {
        std::sort(list.begin(), list.end());
        std::cerr << " " << tag << "=" << std::fixed << std::setprecision(3) << list[list.size() / 2];
    }
    std::cerr << (allCorrect ? "  [correct]" : "  [WRONG RESULT]") << std::endl;
}

namespace {
struct App {
    int order;
    std::string name;
    AppFn fn;
};
std::vector<App>& registry() {
    static std::vector<App> apps;
    return apps;
}
}

Registrar::Registrar(int order, const char* name, AppFn fn) { registry().push_back({order, name, fn}); }

int run(const Config& cfg, const Options& optIn) {
    Options opt = optIn;
    if (opt.threads.empty()) opt.threads.push_back(hardwareThreads());
    const bool gpu = gpu::available();
    std::cerr << "bench: machine=" << opt.machine << " gpu=" << (gpu ? gpu::deviceName() : "none")
              << " cpuThreads=" << hardwareThreads() << " reps=" << opt.reps << std::endl;
    Recorder rec(opt.csv, opt.machine);
    Context ctx{cfg, opt, rec, gpu};
    std::vector<App> apps = registry();
    std::sort(apps.begin(), apps.end(), [](const App& a, const App& b) { return a.order < b.order; });
    if (apps.empty()) std::cerr << "bench: no applications registered yet" << std::endl;
    for (const App& app : apps) {
        if (!opt.apps.empty() && std::find(opt.apps.begin(), opt.apps.end(), app.name) == opt.apps.end()) continue;
        app.fn(ctx);
    }
    return 0;
}
}