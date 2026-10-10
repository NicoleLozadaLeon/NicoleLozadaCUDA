#include <iostream>
#include <string>

#include "Config.h"

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

    // Temporary: the simulation comes in later PRs.
    std::cout << "config OK: " << cfg.nodes.size() << " nodes, " << cfg.streets.size()
              << " streets, " << cfg.restaurants.size() << " restaurants, image " << cfg.map.image
              << ", log " << log_path << "\n";
    return 0;
}
