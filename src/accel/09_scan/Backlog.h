#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "Gpu.h"

// Orders waiting for a courier (concept 09, second application), adapted from
// ec2/09_scan/bikeshare_rentals_in_progress.cu: every order produces two events, +1 when it is created and
// -1 when it is assigned to a courier or rejected. In time order, the inclusive scan of those events is the
// number of orders waiting after each event, and its maximum is the peak of the queue. All values are integers,
// so the result is exact and identical in every run.
namespace backlog {

struct EventStream {                 // the events in time order
    std::vector<int> deltas;         // +1 an order is created, -1 it is assigned or rejected
    std::vector<long long> timesMs;  // when each event happened (to tell the clock time of the peak)
    std::size_t size() const { return deltas.size(); }
};

struct Result {
    std::vector<int> waiting;  // orders waiting for a courier after each event
    int peak = 0;              // the largest value of the curve
    int firstPeak = -1;        // the first event that reached it
};

struct Summary {  // what the final summary shows
    int peak = 0;
    long long peakAtMs = 0;
    int waitingAtEnd = 0;
};

// Reads the events.log that the program writes (one JSON object per line, already in time order).
EventStream loadLog(const std::string& path);

Result cpuOne(const EventStream& events);
// The three-phase scan of 09_task_parallelism/example_54.cpp and exercise_9_2.cpp.
Result cpuMulti(const EventStream& events, int threads);
// The coarsened three-phase scan of the course on the GPU, plus the atomicMax peak and the atomicMin first event.
GpuTiming gpuScan(const EventStream& events, Result& out);

bool equal(const Result& a, const Result& b);
Summary summarize(const EventStream& events, const Result& result);
}
