#pragma once
#include <chrono>
#include <future>
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include "Config.h"
#include "MapGrid.h"
#include "Simulation.h"
class Window {
public:
    explicit Window(const Config& cfg);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    bool frame(const Simulation& sim);
    double fps() const;
    long long frames() const { return frames_; }
private:
    cv::Point toPixel(double lat, double lon) const;
    cv::Point nodePixel(const Graph& graph, int node) const;
    void drawAttribution(cv::Mat& img) const;
    enum class Overlay { None, Demand, Coverage };
    void requestOverlay(const Simulation& sim);
    void collectOverlay();
    const char* overlayName() const;
    const Config& cfg_;
    cv::Mat background_;
    double mercNorth_ = 0.0, mercSouth_ = 0.0;
    const std::string title_ = "Equipetrol Delivery";
    MapGrid grid_;
    Overlay overlay_ = Overlay::None;
    Overlay pendingFor_ = Overlay::None;
    std::future<Grid> pending_;
    cv::Mat overlayImg_;
    cv::Mat overlayMask_;
    std::string snapshot_;
    long long frames_ = 0;
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
};
