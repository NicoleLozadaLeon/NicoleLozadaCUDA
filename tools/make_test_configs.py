#!/usr/bin/env python3
"""Write the test scenarios of section 11 of the brief into config/scenarios/.
usage: python3 tools/make_test_configs.py      (run from the repository root)
Every scenario is config/equipetrol.json with a few values changed.
"""
import copy
import json
import os
SCENARIOS = {
    # name: {section.key: value}   ("restaurants.x" changes every restaurant)
    "fast": {"simulation.timeScale": 600, "simulation.durationS": 1800, "orders.meanIntervalMs": 5000},
    "bursts": {"simulation.timeScale": 600, "simulation.durationS": 1800, "orders.meanIntervalMs": 2000,
               "orders.burstMax": 30},
    "one_slot": {"simulation.timeScale": 600, "simulation.durationS": 1800, "restaurants.pickupSlots": 1},
    "bag_one": {"simulation.timeScale": 600, "simulation.durationS": 1800, "fleet.bagCapacity": 1},
    "breakdowns": {"simulation.timeScale": 300, "simulation.durationS": 0, "orders.meanIntervalMs": 3000,
                   "incidents.breakdownProbability": 1.0},
    "stress": {"simulation.timeScale": 600, "simulation.durationS": 3600, "orders.meanIntervalMs": 2000,
               "orders.burstMax": 30, "orders.maxPending": 20, "fleet.couriers": 16, "fleet.bagCapacity": 1,
               "restaurants.pickupSlots": 1, "incidents.breakdownProbability": 1.0},
    "idle": {"simulation.timeScale": 60, "simulation.durationS": 0, "orders.meanIntervalMs": 1000000000},
}
def main():
    with open("config/equipetrol.json", encoding="utf-8") as f:
        base = json.load(f)
    os.makedirs("config/scenarios", exist_ok=True)
    for name, changes in SCENARIOS.items():
        cfg = copy.deepcopy(base)
        cfg["map"]["image"] = "../../data/equipetrol.png"   # one folder deeper than config/
        for path, value in changes.items():
            section, key = path.split(".")
            if section == "restaurants":
                for r in cfg["restaurants"]:
                    r[key] = value
            else:
                cfg[section][key] = value
        out = f"config/scenarios/{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        print("wrote", out)
if __name__ == "__main__":
    main()
