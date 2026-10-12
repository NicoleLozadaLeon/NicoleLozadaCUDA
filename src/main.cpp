#include <atomic>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>
#include <string>

#include "Config.h"
#include "EventLog.h"
#include "Graph.h"
#include "OrderBook.h"
#include "SimClock.h"

// TEMPORARY: stress test of the EventLog from many threads (removed in PR 3).
static void stress_log(const std::string& path) {
    SimClock clock(60.0);
    EventLog log(path, clock);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&log, t] {
            for (int i = 0; i < 500; ++i) {
                const std::string order = "o" + std::to_string(t * 1000 + i);
                log.orderCreated(order, "r0");
                log.orderDelivered(order, "c" + std::to_string(t));
            }
        });
    }
    for (std::thread& th : threads) th.join();
    std::cout << log.summaryJson() << "\n";
}

// TEMPORARY: 4 producers and 3 consumers hammer one OrderBook(50) (removed in PR 3).
static void stress_book() {
    OrderBook book(50);
    std::atomic<int> accepted{0}, rejected{0}, taken{0};
    std::vector<std::thread> consumers, producers;
    for (int c = 0; c < 3; ++c)
        consumers.emplace_back([&book, &taken] {
            while (std::optional<Order> order = book.take()) taken.fetch_add(1);
        });
    for (int p = 0; p < 4; ++p)
        producers.emplace_back([&book, &accepted, &rejected, p] {
            for (int i = 0; i < 1000; ++i) {
                Order order("o" + std::to_string(p * 1000 + i), 0, 0, 0, 0);
                if (book.submit(std::move(order))) accepted.fetch_add(1);
                else rejected.fetch_add(1);
            }
        });
    for (std::thread& t : producers) t.join();
    book.close();
    for (std::thread& t : consumers) t.join();
    std::cout << "accepted " << accepted << " rejected " << rejected << " taken " << taken
              << " left " << book.size() << "\n";
    std::cout << (accepted + rejected == 4000 && taken + static_cast<int>(book.size()) == accepted ? "OK" : "WRONG")
              << "\n";
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--stress-log") {
        stress_log(argv[2]);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--stress-book") {
        stress_book();
        return 0;
    }
    if (argc < 2) {
        std::cerr << "usage: delivery_sim <config.json> [--log <path>]\n";
        return 1;
    }
    std::string log_path = "events.log";
    if (argc == 4 && std::string(argv[2]) == "--log") {
        log_path = argv[3];
    } else if (argc != 2) {
        std::cerr << "usage: delivery_sim <config.json> [--log <path>]\n";
        return 1;
    }

    Config cfg;
    try {
        cfg = load_config(argv[1]);
    } catch (const ConfigError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    Graph graph(cfg);
    int unreachable_pairs = 0;
    for (int a = 0; a < graph.nodeCount(); ++a)
        for (int b = 0; b < graph.nodeCount(); ++b)
            if (!graph.reachable(a, b)) ++unreachable_pairs;
    std::cout << "graph: " << graph.nodeCount() << " nodes, unreachable pairs: " << unreachable_pairs << "\n";

    // Temporary: the simulation comes in PR 3.
    std::cout << "config OK: " << cfg.nodes.size() << " nodes, " << cfg.streets.size()
              << " streets, " << cfg.restaurants.size() << " restaurants, image " << cfg.map.image
              << ", log " << log_path << "\n";
    return 0;
}
