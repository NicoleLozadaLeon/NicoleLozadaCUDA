#pragma once
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include "Config.h"
#include "Simulation.h"
// The live view (DP.8). It runs on the main thread, which OpenCV's highgui expects.
// It only READS the simulation, through snapshots taken under each monitor's own lock,
// one lock at a time, so drawing can never block a courier for long or deadlock.
class Window {
public:
    // Loads the map image; throws std::runtime_error if it cannot be read.
    explicit Window(const Config& cfg);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    // Draws one frame, then waits for the next one: this pause only paces the frame rate.
    // Returns false when the user closes the window or presses q or Esc.
    bool frame(const Simulation& sim);
private:
    cv::Point toPixel(double lat, double lon) const;
    cv::Point nodePixel(const Graph& graph, int node) const;
    void drawAttribution(cv::Mat& img) const;
    const Config& cfg_;
    cv::Mat background_;   // map + streets + restaurants, drawn once
    double mercNorth_ = 0.0, mercSouth_ = 0.0;
    const std::string title_ = "Equipetrol Delivery";
};