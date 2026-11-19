#pragma once
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                  \
    do {                                                                                                  \
        const cudaError_t err_ = (call);                                                                  \
        if (err_ != cudaSuccess)                                                                          \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " at " +   \
                                     __FILE__ + ":" + std::to_string(__LINE__));                          \
    } while (0)

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(size_, other.size_);
        return *this;
    }
    void resize(std::size_t n) {
        if (n == size_) return;
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = 0;
        if (n > 0) CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        size_ = n;
    }
    void upload(const T* host, std::size_t n) {
        resize(n);
        if (n > 0) CUDA_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, std::size_t n) const {
        if (n > 0) CUDA_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (size_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }
    T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

class KernelTimer {
public:
    KernelTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~KernelTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    KernelTimer(const KernelTimer&) = delete;
    KernelTimer& operator=(const KernelTimer&) = delete;
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    double stop() {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

inline double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

inline unsigned blocksFor(std::size_t n, unsigned threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}