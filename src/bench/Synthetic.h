#pragma once
#include <string>
#include <vector>
#include "MapGrid.h"

namespace bench {
std::vector<int> deliveryTimes(long long n, unsigned seed);
std::vector<int> restaurantKeys(long long n, int restaurants, unsigned seed);
Grid scatter(const MapGrid& map, int points, int hotspots, unsigned seed);
std::string gridLabel(double cellM, const Grid& g);
}