#include "Backlog.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include "Parallel.h"

namespace backlog {

EventStream loadLog(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open the event log: " + path);
    EventStream events;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t t = line.find("\"t\":");
        const std::size_t e = line.find("\"event\":\"");
        if (t == std::string::npos || e == std::string::npos) continue;
        const std::size_t from = e + 9, to = line.find('"', from);
        if (to == std::string::npos) continue;
        const std::string event = line.substr(from, to - from);
        int delta = 0;
        if (event == "orderCreated") delta = +1;
        else if (event == "orderAssigned" || event == "orderRejected") delta = -1;
        if (delta == 0) continue;  // the other events do not change the queue
        events.deltas.push_back(delta);
        events.timesMs.push_back(std::atoll(line.c_str() + t + 4));
    }
    return events;
}

// Reference: one running sum and the first event of the maximum.
Result cpuOne(const EventStream& events) {
    Result r;
    r.waiting.resize(events.size());
    int run = 0, peak = INT_MIN, first = -1;
    for (std::size_t i = 0; i < events.size(); ++i) {
        run += events.deltas[i];
        r.waiting[i] = run;
        if (run > peak) {
            peak = run;
            first = static_cast<int>(i);
        }
    }
    r.peak = events.size() ? peak : 0;
    r.firstPeak = first;
    return r;
}

// Phase 1: each thread scans its own range and records its total and its local peak. Phase 2 (tiny, sequential):
// an exclusive scan of the totals gives the offset of every range, and the global peak is the best local peak
// plus its offset. Phase 3: each thread adds the offset of its range.
Result cpuMulti(const EventStream& events, int threads) {
    const std::size_t n = events.size(), t = static_cast<std::size_t>(std::max(1, threads));
    Result r;
    r.waiting.resize(n);
    if (n == 0) return r;
    std::vector<int> total(t, 0), localMax(t, INT_MIN), localFirst(t, -1), offset(t, 0);
    parallelFor(n, static_cast<int>(t), [&](std::size_t begin, std::size_t end, std::size_t id) {
        int run = 0;
        for (std::size_t i = begin; i < end; ++i) {
            run += events.deltas[i];
            r.waiting[i] = run;
            if (run > localMax[id]) {
                localMax[id] = run;
                localFirst[id] = static_cast<int>(i);
            }
        }
        total[id] = run;
    });
    int peak = INT_MIN, first = -1;
    for (std::size_t c = 0; c < t; ++c) {
        if (c > 0) offset[c] = offset[c - 1] + total[c - 1];
        if (localFirst[c] < 0) continue;
        const int candidate = localMax[c] + offset[c];
        if (candidate > peak) {  // strictly greater: the earliest range wins a tie
            peak = candidate;
            first = localFirst[c];
        }
    }
    parallelFor(n, static_cast<int>(t), [&](std::size_t begin, std::size_t end, std::size_t id) {
        if (offset[id] == 0) return;
        for (std::size_t i = begin; i < end; ++i) r.waiting[i] += offset[id];
    });
    r.peak = peak;
    r.firstPeak = first;
    return r;
}

bool equal(const Result& a, const Result& b) {
    return a.peak == b.peak && a.firstPeak == b.firstPeak && a.waiting == b.waiting;
}

Summary summarize(const EventStream& events, const Result& result) {
    Summary s;
    if (events.size() == 0 || result.firstPeak < 0) return s;
    s.peak = result.peak;
    s.peakAtMs = events.timesMs[static_cast<std::size_t>(result.firstPeak)];
    s.waitingAtEnd = result.waiting.back();
    return s;
}

#ifndef EQ_USE_CUDA
GpuTiming gpuScan(const EventStream&, Result&) { throw std::runtime_error("built without CUDA"); }
#endif
}
