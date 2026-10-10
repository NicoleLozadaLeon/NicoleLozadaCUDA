#include <iostream>
#include <string>

#include "Config.h"
#include "Graph.h"

int main(int argc, char** argv) {
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
    const int start = graph.indexOf(cfg.fleet.startNode);
    for (const RestaurantDef& r : cfg.restaurants) {
        const int to = graph.indexOf(r.node);
        std::cout << "  " << r.name << ": " << static_cast<int>(graph.distance(start, to)) << " m, "
                  << graph.route(start, to).size() << " nodes\n";
    }
    for (const StreetDef& st : cfg.streets) {
        if (st.oneWay) {
            const int a = graph.indexOf(st.from), b = graph.indexOf(st.to);
            std::cout << "  one-way " << st.id << ": " << static_cast<int>(graph.distance(a, b)) << " m forward, "
                      << static_cast<int>(graph.distance(b, a)) << " m back\n";
            break;
        }
    }

    // Temporary: the simulation comes in later PRs.
    std::cout << "config OK: " << cfg.nodes.size() << " nodes, " << cfg.streets.size()
              << " streets, " << cfg.restaurants.size() << " restaurants, image " << cfg.map.image
              << ", log " << log_path << "\n";
    return 0;
}
