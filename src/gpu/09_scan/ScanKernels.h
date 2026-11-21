#pragma once

namespace scan {
long long scratchFor(long long n);
void exclusiveScan(unsigned* data, long long n, unsigned* scratch);
}
