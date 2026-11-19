#include "Argmin.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "Parallel.h"

namespace argmin {
namespace {
void rowsMin(const std::vector<int>& eta, int cols, const std::vector<int>& ids, std::size_t begin,
             std::size_t end, std::vector<Key>& out) {
    const std::size_t c = static_cast<std::size_t>(cols);
    for (std::size_t r = begin; r < end; ++r) {
        Key best = kNone;
        for (std::size_t j = 0; j < c; ++j) best = std::min(best, pack(eta[r * c + j], ids[j]));
        out[r] = best;
    }
}
}

std::vector<Key> cpuOne(const std::vector<int>& eta, int rows, int cols, const std::vector<int>& ids) {
    std::vector<Key> out(static_cast<std::size_t>(rows));
    rowsMin(eta, cols, ids, 0, out.size(), out);
    return out;
}

std::vector<Key> cpuMulti(const std::vector<int>& eta, int rows, int cols, const std::vector<int>& ids,
                          int threads) {
    std::vector<Key> out(static_cast<std::size_t>(rows));
    parallelFor(out.size(), threads, [&](std::size_t begin, std::size_t end, std::size_t) {
        rowsMin(eta, cols, ids, begin, end, out);
    });
    return out;
}

#ifndef EQ_USE_CUDA
GpuTiming gpu(const std::vector<int>&, int, int, const std::vector<int>&, std::vector<Key>&) {
    throw std::runtime_error("built without CUDA");
}
Key bestCourierGpu(const eta::DeviceTable&, const eta::Params&, const eta::Query&, const eta::Couriers&,
                   const std::vector<int>&) {
    throw std::runtime_error("built without CUDA");
}
#endif
}
