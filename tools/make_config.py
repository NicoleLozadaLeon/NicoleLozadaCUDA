#!/usr/bin/env python3
"""Builds config/equipetrol.json from the traced network (tools/network_draft.json)."""
import json, math

DRAFT = "tools/network_draft.json"
OUT = "config/equipetrol.json"
BOUNDS = {"north": -17.752162388363526, "south": -17.779011638450203,
          "west": -63.20374411309951, "east": -63.18915387798629}
SLOTS = [1, 2, 1, 2, 1, 2, 1, 2]          # pickupSlots per restaurant, in order
PREP_MS = [60000, 240000]                 # [min, max] preparation time

def dump(cfg, path):
    def arr(items):
        return "[\n" + ",\n".join("    " + json.dumps(x, ensure_ascii=False) for x in items) + "\n  ]"
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write('  "map": ' + json.dumps(cfg["map"], ensure_ascii=False) + ",\n")
        f.write('  "nodes": ' + arr(cfg["nodes"]) + ",\n")
        f.write('  "streets": ' + arr(cfg["streets"]) + ",\n")
        f.write('  "restaurants": ' + arr(cfg["restaurants"]) + ",\n")
        for k in ("fleet", "orders", "dispatch", "incidents"):
            f.write(f'  "{k}": ' + json.dumps(cfg[k]) + ",\n")
        f.write('  "simulation": ' + json.dumps(cfg["simulation"]) + "\n}\n")

def main():
    with open(DRAFT, encoding="utf-8") as f:
        d = json.load(f)
    nodes, streets = d["nodes"], d["streets"]
    clat = sum(n["lat"] for n in nodes) / len(nodes)
    clon = sum(n["lon"] for n in nodes) / len(nodes)
    start = min(nodes, key=lambda n: (n["lat"] - clat) ** 2 + (n["lon"] - clon) ** 2)["id"]
    restaurants = []
    for i, r in enumerate(d["restaurants"]):
        restaurants.append({"id": f"r{i}", "name": r["name"], "node": r["node"],
                            "pickupSlots": SLOTS[i % len(SLOTS)], "prepTimeMs": PREP_MS})
    cfg = {
        "map": {"image": "../data/equipetrol.jpg",
                "attribution": "© OpenStreetMap contributors", "bounds": BOUNDS},
        "nodes": nodes, "streets": streets, "restaurants": restaurants,
        "fleet": {"couriers": 8, "bagCapacity": 3, "speedKmh": 30, "startNode": start},
        "orders": {"meanIntervalMs": 20000, "burstMax": 12, "maxPending": 50, "seed": 42},
        "dispatch": {"quoteTimeoutMs": 200, "acceptTimeoutMs": 600000},
        "incidents": {"breakdownProbability": 0.02},
        "simulation": {"durationS": 3600, "timeScale": 60},
    }
    dump(cfg, OUT)
    print(f"wrote {OUT}: {len(nodes)} nodes, {len(streets)} streets, {len(restaurants)} restaurants, start {start}")

if __name__ == "__main__":
    main()
