#pragma once
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include "Config.h"
#include "Simulation.h"
class Window {
public:
    explicit Window(const Config& cfg);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    bool frame(const Simulation& sim);
private:
    cv::Point toPixel(double lat, double lon) const;
    cv::Point nodePixel(const Graph& graph, int node) const;
    void drawAttribution(cv::Mat& img) const;
    const Config& cfg_;
    cv::Mat background_;
    double mercNorth_ = 0.0, mercSouth_ = 0.0;
    const std::string title_ = "Equipetrol Delivery";
};
