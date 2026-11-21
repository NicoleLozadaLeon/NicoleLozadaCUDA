#include "Window.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
namespace {
const double kPi = 3.14159265358979323846;
const int kFrameMs = 33;
const double kMaxHeight = 900.0;
double mercator(double latDeg) {
    return std::log(std::tan(kPi / 4.0 + latDeg * kPi / 360.0));
}
cv::Scalar colorOf(Courier::State state) {
    switch (state) {
        case Courier::State::ToRestaurant: return cv::Scalar(205, 110, 40);
        case Courier::State::Carrying: return cv::Scalar(70, 160, 40);
        case Courier::State::Broken: return cv::Scalar(40, 40, 215);
        case Courier::State::Idle: break;
    }
    return cv::Scalar(130, 130, 130);
}
void text(cv::Mat& img, const std::string& s, cv::Point at, double size = 0.45) {
    cv::putText(img, s, at, cv::FONT_HERSHEY_SIMPLEX, size, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
}
}
Window::Window(const Config& cfg) : cfg_(cfg), grid_(cfg.map.bounds, 25.0) {
    cv::setNumThreads(0);
    if (const char* v = std::getenv("EQ_OVERLAY")) {
        const std::string name(v);
        if (name == "demand") overlay_ = Overlay::Demand;
        else if (name == "coverage") overlay_ = Overlay::Coverage;
    }
    if (const char* v = std::getenv("EQ_SNAPSHOT")) snapshot_ = v;

    const cv::Mat image = cv::imread(cfg.map.image, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot read the map image: " + cfg.map.image);
    const double scale = std::min(1.0, kMaxHeight / image.rows);
    cv::resize(image, background_, cv::Size(), scale, scale, cv::INTER_AREA);
    mercNorth_ = mercator(cfg.map.bounds.north);
    mercSouth_ = mercator(cfg.map.bounds.south);
    std::unordered_map<std::string, cv::Point> at;
    for (const NodeDef& n : cfg.nodes) at[n.id] = toPixel(n.lat, n.lon);
    for (const StreetDef& s : cfg.streets)
        cv::line(background_, at[s.from], at[s.to], cv::Scalar(120, 120, 120), 2, cv::LINE_AA);
    for (const RestaurantDef& r : cfg.restaurants) {
        const cv::Point p = at[r.node];
        cv::rectangle(background_, p - cv::Point(6, 6), p + cv::Point(6, 6), cv::Scalar(20, 20, 20), cv::FILLED);
        text(background_, r.name, p + cv::Point(9, 4), 0.4);
    }
    cv::namedWindow(title_, cv::WINDOW_AUTOSIZE);
}
Window::~Window() { cv::destroyAllWindows(); }
cv::Point Window::toPixel(double lat, double lon) const {
    const Bounds& b = cfg_.map.bounds;
    const double x = (lon - b.west) / (b.east - b.west) * background_.cols;
    const double y = (mercNorth_ - mercator(lat)) / (mercNorth_ - mercSouth_) * background_.rows;
    return cv::Point(cvRound(x), cvRound(y));
}
cv::Point Window::nodePixel(const Graph& graph, int node) const {
    return toPixel(graph.lat(node), graph.lon(node));
}
bool Window::frame(const Simulation& sim) {
    cv::Mat img = background_.clone();
    if (overlay_ != Overlay::None && frames_ % 30 == 0) requestOverlay(sim);
    collectOverlay();
    if (overlay_ != Overlay::None && !overlayImg_.empty()) {
        cv::Mat blended;
        cv::addWeighted(img, 0.45, overlayImg_, 0.55, 0.0, blended);
        blended.copyTo(img, overlayMask_);
    }
    const Graph& graph = sim.graph();
    const long long now = sim.clock().nowMs();
    std::vector<Courier::View> views;
    for (const std::shared_ptr<Courier>& c : sim.fleet().couriers()) views.push_back(c->view());
    for (int node : sim.book().deliveryNodes())
        cv::circle(img, nodePixel(graph, node), 5, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    for (const Courier::View& v : views)
        for (const Courier::View::Item& item : v.bag)
            cv::circle(img, nodePixel(graph, item.deliveryNode), 5, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    for (const Courier::View& v : views) {
        double f = 1.0;
        if (v.legFrom != v.legTo && v.legEndMs > v.legStartMs)
            f = std::clamp(static_cast<double>(now - v.legStartMs) / static_cast<double>(v.legEndMs - v.legStartMs),
                           0.0, 1.0);
        const double lat = graph.lat(v.legFrom) + (graph.lat(v.legTo) - graph.lat(v.legFrom)) * f;
        const double lon = graph.lon(v.legFrom) + (graph.lon(v.legTo) - graph.lon(v.legFrom)) * f;
        const cv::Point p = toPixel(lat, lon);
        cv::circle(img, p, 8, colorOf(v.state), cv::FILLED, cv::LINE_AA);
        cv::circle(img, p, 8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    const EventLog::Summary s = sim.log().summary();
    cv::rectangle(img, cv::Rect(8, 8, 300, 128), cv::Scalar(250, 250, 250), cv::FILLED);
    cv::rectangle(img, cv::Rect(8, 8, 300, 128), cv::Scalar(60, 60, 60), 1);
    text(img, "created " + std::to_string(s.created) + "  delivered " + std::to_string(s.delivered) +
                  "  rejected " + std::to_string(s.rejected), cv::Point(16, 28));
    text(img, "pending " + std::to_string(s.pending) + "  reassigned " + std::to_string(s.reassigned) +
                  "  breakdowns " + std::to_string(s.breakdowns), cv::Point(16, 48));
    text(img, "avg delivery " + std::to_string(s.avgDeliveryMs / 60000) + " min  max " +
                  std::to_string(s.maxDeliveryMs / 60000) + " min", cv::Point(16, 68));
    text(img, "sim time " + std::to_string(now / 60000) + " min   late/failed estimates " +
                  std::to_string(sim.dispatcher().late()) + "/" + std::to_string(sim.dispatcher().failed()),
         cv::Point(16, 88), 0.4);
    const char* labels[] = {"idle", "to restaurant", "carrying", "broken"};
    const Courier::State states[] = {Courier::State::Idle, Courier::State::ToRestaurant, Courier::State::Carrying,
                                     Courier::State::Broken};
    for (int i = 0; i < 4; ++i) {
        const cv::Point p(20 + i * 72, 116);
        cv::circle(img, p, 6, colorOf(states[i]), cv::FILLED, cv::LINE_AA);
        text(img, labels[i], p + cv::Point(9, 4), 0.33);
    }
    cv::rectangle(img, cv::Rect(8, 140, 300, 22), cv::Scalar(250, 250, 250), cv::FILLED);
    text(img, std::string("overlay: ") + overlayName() + "  (press o to change)", cv::Point(16, 156), 0.4);
    drawAttribution(img);
    cv::imshow(title_, img);
    if (!snapshot_.empty() && frames_ % 150 == 149) cv::imwrite(snapshot_, img);
    ++frames_;
    const int key = cv::waitKey(kFrameMs);
    if (key == 'q' || key == 27) return false;
    if (key == 'o') {
        overlay_ = static_cast<Overlay>((static_cast<int>(overlay_) + 1) % 3);
        overlayImg_.release();
        frames_ -= frames_ % 30;
    }
    return cv::getWindowProperty(title_, cv::WND_PROP_VISIBLE) >= 1;
}
void Window::drawAttribution(cv::Mat& img) const {
    std::string attribution = cfg_.map.attribution;
    const std::string copyright = "\xC2\xA9";
    const bool symbol = attribution.compare(0, copyright.size(), copyright) == 0;
    if (symbol) attribution = attribution.substr(copyright.size());
    int baseline = 0;
    const cv::Size size = cv::getTextSize(attribution, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    const int x = img.cols - size.width - (symbol ? 26 : 10);
    const int y = img.rows - 10;
    cv::rectangle(img, cv::Rect(x - 6, y - size.height - 6, img.cols - x, size.height + 12), cv::Scalar(250, 250, 250),
                  cv::FILLED);
    if (symbol) {
        cv::circle(img, cv::Point(x + 6, y - 5), 7, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);
        text(img, "c", cv::Point(x + 2, y - 1), 0.4);
    }
    text(img, attribution, cv::Point(symbol ? x + 16 : x, y));
}
double Window::fps() const {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
    return s > 0.0 ? static_cast<double>(frames_) / s : 0.0;
}
const char* Window::overlayName() const {
    switch (overlay_) {
        case Overlay::Demand: return "demand heatmap";
        case Overlay::Coverage: return "courier coverage (red = poorly covered)";
        case Overlay::None: break;
    }
    return "none";
}
void Window::requestOverlay(const Simulation& sim) {
    if (pending_.valid()) return;
    const Graph& graph = sim.graph();
    Grid g = grid_.empty();
    if (overlay_ == Overlay::Coverage) {
        for (const std::shared_ptr<Courier>& c : sim.fleet().couriers()) {
            const Courier::View v = c->view();
            grid_.add(g, graph.lat(v.legTo), graph.lon(v.legTo));
        }
        pending_ = sim.accel().coverage(std::move(g), 100);
    } else {
        const long long now = sim.clock().nowMs();
        for (const EventLog::Delivery& d : sim.log().deliveries())
            if (now - d.deliveredAtMs < 3600000) grid_.add(g, graph.lat(d.deliveryNode), graph.lon(d.deliveryNode));
        for (int node : sim.book().deliveryNodes()) grid_.add(g, graph.lat(node), graph.lon(node));
        pending_ = sim.accel().smooth(std::move(g), 2);
    }
    pendingFor_ = overlay_;
}
void Window::collectOverlay() {
    if (!pending_.valid() || pending_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    Grid g;
    try {
        g = pending_.get();
    } catch (const std::exception& e) {
        std::cerr << "window: overlay failed: " << e.what() << "\n";
        overlay_ = Overlay::None;
        return;
    }
    if (pendingFor_ != overlay_ || g.v.empty()) return;
    float peak = 0.0f;
    for (float v : g.v) peak = std::max(peak, v);
    cv::Mat values(g.h, g.w, CV_32F, g.v.data());
    if (peak > 0.0f) values /= peak;
    if (overlay_ == Overlay::Coverage) cv::pow(values, 0.25, values);
    cv::Mat scaled;
    values.convertTo(scaled, CV_8U, 255.0);
    if (overlay_ == Overlay::Coverage) scaled = 255 - scaled;
    cv::Mat colored;
    cv::applyColorMap(scaled, colored, cv::COLORMAP_JET);
    cv::resize(colored, overlayImg_, background_.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat mask = overlay_ == Overlay::Coverage ? cv::Mat(scaled.size(), CV_8U, cv::Scalar(255)) : cv::Mat(scaled > 12);
    cv::resize(mask, overlayMask_, background_.size(), 0, 0, cv::INTER_NEAREST);
}
