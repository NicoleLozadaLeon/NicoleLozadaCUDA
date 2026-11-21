#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include "Config.h"

struct Grid {
    int w = 0, h = 0;
    std::vector<float> v;
    Grid() = default;
    Grid(int width, int height) : w(width), h(height), v(static_cast<std::size_t>(width) * height, 0.0f) {}
    float& at(int x, int y) { return v[static_cast<std::size_t>(y) * w + x]; }
    float at(int x, int y) const { return v[static_cast<std::size_t>(y) * w + x]; }
};

class MapGrid {
public:
    MapGrid(const Bounds& b, double cellM) : bounds_(b), cellM_(cellM) {
        const double midLat = (b.north + b.south) / 2.0 * 3.14159265358979323846 / 180.0;
        mPerLon_ = 111320.0 * std::cos(midLat);
        mPerLat_ = 110540.0;
        w_ = static_cast<int>(std::ceil((b.east - b.west) * mPerLon_ / cellM));
        h_ = static_cast<int>(std::ceil((b.north - b.south) * mPerLat_ / cellM));
    }
    int width() const { return w_; }
    int height() const { return h_; }
    Grid empty() const { return Grid(w_, h_); }
    bool cellOf(double lat, double lon, int& x, int& y) const {
        x = static_cast<int>((lon - bounds_.west) * mPerLon_ / cellM_);
        y = static_cast<int>((bounds_.north - lat) * mPerLat_ / cellM_);
        return x >= 0 && y >= 0 && x < w_ && y < h_;
    }
    void add(Grid& g, double lat, double lon, float weight = 1.0f) const {
        int x = 0, y = 0;
        if (cellOf(lat, lon, x, y)) g.at(x, y) += weight;
    }

private:
    Bounds bounds_;
    double cellM_;
    double mPerLon_ = 0.0, mPerLat_ = 0.0;
    int w_ = 0, h_ = 0;
};

inline double maxRelativeError(const Grid& a, const Grid& ref) {
    double peak = 0.0, worst = 0.0;
    for (float r : ref.v) peak = std::fmax(peak, std::fabs(r));
    if (a.v.size() != ref.v.size()) return 1e30;
    for (std::size_t i = 0; i < ref.v.size(); ++i) worst = std::fmax(worst, std::fabs(a.v[i] - ref.v[i]));
    return peak > 0.0 ? worst / peak : worst;
}