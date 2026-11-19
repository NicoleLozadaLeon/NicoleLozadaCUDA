#include "Synthetic.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <random>
#include <utility>

namespace bench {
std::vector<int> deliveryTimes(long long n, unsigned seed) {
    std::mt19937 rng(seed);
    std::gamma_distribution<double> minutes(4.0, 7.0);
    std::vector<int> ms(static_cast<std::size_t>(n));
    for (int& v : ms) v = static_cast<int>(std::min(minutes(rng), 1400.0) * 60000.0);
    return ms;
}

std::vector<int> restaurantKeys(long long n, int restaurants, unsigned seed) {
    std::mt19937 rng(seed);
    std::bernoulli_distribution popular(0.5);
    std::uniform_int_distribution<int> top(0, std::min(7, restaurants - 1)), any(0, restaurants - 1);
    std::vector<int> keys(static_cast<std::size_t>(n));
    for (int& k : keys) k = popular(rng) ? top(rng) : any(rng);
    return keys;
}

Grid scatter(const MapGrid& map, int points, int hotspots, unsigned seed) {
    std::mt19937 rng(seed);
    Grid g = map.empty();
    std::uniform_int_distribution<int> x(0, g.w - 1), y(0, g.h - 1);
    std::normal_distribution<double> spread(0.0, g.w / 20.0 + 1.0);
    std::vector<std::pair<int, int>> centers;
    for (int h = 0; h < hotspots; ++h) centers.push_back({x(rng), y(rng)});
    for (int i = 0; i < points; ++i) {
        int px = x(rng), py = y(rng);
        if (!centers.empty()) {
            const auto& c = centers[static_cast<std::size_t>(i) % centers.size()];
            px = c.first + static_cast<int>(spread(rng));
            py = c.second + static_cast<int>(spread(rng));
        }
        if (px >= 0 && py >= 0 && px < g.w && py < g.h) g.at(px, py) += 1.0f;
    }
    return g;
}

std::string gridLabel(double cellM, const Grid& g) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%gm %dx%d", cellM, g.w, g.h);
    return buf;
}
}