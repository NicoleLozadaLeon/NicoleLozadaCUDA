#pragma once
#include "Eta.h"

namespace eta {
void launchSoA(const DeviceTable& table, const Params& p, const Query* orders, int m, const int* freeNode,
               const long long* freeAtMs, int k, int* out);
}
