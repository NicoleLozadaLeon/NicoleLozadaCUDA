#pragma once
#include <chrono>

// Simulated time: simulated ms = real ms * timeScale.
// Every member is const, so any thread can read it without a lock.
class SimClock {
public:
    using Clock = std::chrono::steady_clock;

    explicit SimClock(double timeScale) : start_(Clock::now()), scale_(timeScale) {}

    double scale() const { return scale_; }
    long long nowMs() const { return toSimMs(Clock::now()); }

    long long toSimMs(Clock::time_point real) const {
        const std::chrono::duration<double, std::milli> elapsed = real - start_;
        return static_cast<long long>(elapsed.count() * scale_);
    }

    Clock::time_point toReal(long long simMs) const {
        const std::chrono::duration<double, std::milli> real(static_cast<double>(simMs) / scale_);
        return start_ + std::chrono::duration_cast<Clock::duration>(real);
    }

private:
    const Clock::time_point start_;
    const double scale_;
};
