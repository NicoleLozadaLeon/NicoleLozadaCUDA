#pragma once
#include <string>

struct GpuTiming {
    double kernelMs = 0.0;
    double e2eMs = 0.0;
};

namespace gpu {
bool compiled();
bool available();
bool start();
std::string deviceName();
int smCount();
}