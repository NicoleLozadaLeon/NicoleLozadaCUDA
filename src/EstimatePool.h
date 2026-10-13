#pragma once
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
// A fixed set of worker threads fed from a monitor queue (mutex + condition variable).
// submit() wraps each job in a std::packaged_task, so the caller gets a std::future with the
// result or the exception. Unlike a std::async future, a packaged_task future does NOT block
// in its destructor, so the dispatcher can walk away from a late estimate.
class EstimatePool {
public:
    explicit EstimatePool(std::size_t workers) {
        for (std::size_t i = 0; i < workers; ++i) workers_.emplace_back(&EstimatePool::workerLoop, this);
    }
    ~EstimatePool() { shutdown(); }
    EstimatePool(const EstimatePool&) = delete;
    EstimatePool& operator=(const EstimatePool&) = delete;
    template <class Fn>
    auto submit(Fn fn) -> std::future<decltype(fn())> {
        using R = decltype(fn());
        // packaged_task is move-only; the queue stores copyable std::function, so share it.
        auto task = std::make_shared<std::packaged_task<R()>>(std::move(fn));
        std::future<R> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push([task] { (*task)(); });
        }
        cond_.notify_one();
        return result;
    }
    // Wakes every worker and joins them all. Jobs still queued are dropped: the simulation is
    // stopping and nobody waits for them (the pool of the course drains them instead).
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return;
            stopping_ = true;
        }
        cond_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }
    std::size_t size() const { return workers_.size(); }
private:
    void workerLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
                if (stopping_) return;
                job = std::move(jobs_.front());
                jobs_.pop();
            }
            job();   // run OUTSIDE the lock
        }
    }
    std::vector<std::thread> workers_;

    std::queue<std::function<void()>> jobs_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool stopping_ = false;
};