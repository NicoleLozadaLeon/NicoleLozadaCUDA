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
        auto task = std::make_shared<std::packaged_task<R()>>(std::move(fn));
        std::future<R> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push([task] { (*task)(); });
        }
        cond_.notify_one();
        return result;
    }
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
            job();
        }
    }
    std::vector<std::thread> workers_;

    std::queue<std::function<void()>> jobs_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool stopping_ = false;
};
