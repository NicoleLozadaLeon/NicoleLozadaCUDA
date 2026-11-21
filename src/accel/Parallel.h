#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

class Barrier {
public:
    explicit Barrier(std::size_t parties) : parties_(parties) {}
    void arriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::size_t generation = generation_;
        if (++arrived_ == parties_) {
            arrived_ = 0;
            ++generation_;
            lock.unlock();
            cond_.notify_all();
            return;
        }
        cond_.wait(lock, [&] { return generation_ != generation; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cond_;
    const std::size_t parties_;
    std::size_t arrived_ = 0;
    std::size_t generation_ = 0;
};

inline int hardwareThreads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 2 : static_cast<int>(hw);
}

template <class Fn>
void parallelFor(std::size_t n, int threads, Fn fn) {
    const std::size_t t = static_cast<std::size_t>(std::max(1, threads));
    const std::size_t chunk = (n + t - 1) / t;
    std::vector<std::future<void>> parts;
    for (std::size_t i = 0; i < t; ++i) {
        const std::size_t begin = i * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        if (begin >= end) break;
        parts.push_back(std::async(std::launch::async, fn, begin, end, i));
    }
    for (std::future<void>& p : parts) p.get();
}