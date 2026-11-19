#include "Gpu.h"
#include "Cuda.h"

namespace gpu {
bool compiled() { return true; }
bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
bool start() { return available() && cudaSetDevice(0) == cudaSuccess && cudaFree(nullptr) == cudaSuccess; }
std::string deviceName() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "none";
    return prop.name;
}
int smCount() {
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    return sms;
}
}