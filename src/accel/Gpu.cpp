#include "Gpu.h"

#ifndef EQ_USE_CUDA
namespace gpu {
bool compiled() { return false; }
bool available() { return false; }
bool start() { return false; }
std::string deviceName() { return "none (built without CUDA)"; }
int smCount() { return 0; }
}
#endif