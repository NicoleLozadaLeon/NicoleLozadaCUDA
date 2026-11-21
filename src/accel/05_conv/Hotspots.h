#pragma once
#include <vector>
#include "Gpu.h"
#include "MapGrid.h"

// Where is the pending demand, and which zones have no courier near them?
// The unresolved orders are counted on a grid, the counts are smoothed with a 13 x 13 Gaussian (sigma = 3 cells)
// and the strongest zones are ranked, each with its nearest courier.
namespace hotspots {
constexpr int kRadius = 6;
constexpr int kSide = 2 * kRadius + 1;           // 13 x 13 weights
constexpr double kSigmaCells = 3.0;
constexpr double kTolerance = 1e-5;              // maximum error relative to the maximum of the map
constexpr int kZones = 5;                        // zones in the ranking
constexpr int kMinSeparationCells = 15;          // two zones are never closer than this
constexpr double kFarMetres = 500.0;             // a zone with its nearest courier farther than this is uncovered

struct Cell {
    int x = 0, y = 0;
};

struct Hotspot {
    int x = 0, y = 0;
    float value = 0.0f;       // smoothed orders per cell
    int courier = -1;         // index of the nearest courier, -1 when there is none
    double distanceM = 0.0;
    bool uncovered = true;
};

std::vector<float> gaussian();
// `cells` holds the cell number (y * w + x) of every pending order.
Grid countPoints(int w, int h, const std::vector<int>& cells);
Grid cpuOne(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter);
Grid cpuMulti(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, int threads);
// Counting with atomicAdd and the smoothing with a shared-memory tile: 32 x 32 threads load, 20 x 20 write.
GpuTiming gpuTiled(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, Grid& out);
// The same two steps, but the smoothing reads the grid straight from global memory: 16 x 16 threads, all write.
GpuTiming gpuDirect(int w, int h, const std::vector<int>& cells, const std::vector<float>& filter, Grid& out);

// Keeps the GPU busy for `ms` milliseconds so it leaves its idle clock. Only the benchmark calls it, before a
// timed run: a laptop GPU that waited a second for the CPU variants starts the next kernel at 210 MHz.
void gpuWarmUp(int ms);

// The strongest zones of a smoothed map, in order, each one with its nearest courier. Ties go to the lowest cell.
std::vector<Hotspot> rank(const Grid& smoothed, double cellM, const std::vector<Cell>& couriers,
                          int count = kZones, int minSeparationCells = kMinSeparationCells);
}
