#!/usr/bin/env python3
"""Write a city-scale configuration: an N x N street grid over the Equipetrol map, for the GPU work (EC2).

usage: python3 tools/make_city_config.py <side> <couriers> <restaurants> <meanIntervalMs> <durationS> <out.json>
example: python3 tools/make_city_config.py 40 400 24 300 3600 config/city/city_40.json
Every third row is a one-way street (alternating direction); every column is two-way, so the grid stays
strongly connected. Restaurants are spread over the grid; durationS is in simulated seconds (timeScale 60: 3600 = one simulated hour = 60 real seconds).
"""
import json
import os
import random
import sys


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    side, couriers, restaurants, interval, duration = (int(a) for a in sys.argv[1:6])
    out = sys.argv[6]
    with open("config/equipetrol.json", encoding="utf-8") as f:
        base = json.load(f)
    b = base["map"]["bounds"]
    rng = random.Random(42)
    nodes, streets = [], []
    for r in range(side):
        for c in range(side):
            lat = b["north"] + (b["south"] - b["north"]) * (r + 0.5) / side
            lon = b["west"] + (b["east"] - b["west"]) * (c + 0.5) / side
            nodes.append({"id": f"g{r}_{c}", "lat": lat, "lon": lon})
    for r in range(side):
        one_way = r % 3 == 1
        eastward = (r // 3) % 2 == 0
        for c in range(side - 1):
            a, z = f"g{r}_{c}", f"g{r}_{c + 1}"
            if one_way and not eastward:
                a, z = z, a
            streets.append({"id": f"h{r}_{c}", "from": a, "to": z, "oneWay": one_way})
    for r in range(side - 1):
        for c in range(side):
            streets.append({"id": f"v{r}_{c}", "from": f"g{r}_{c}", "to": f"g{r + 1}_{c}", "oneWay": False})
    cells = rng.sample(range(side * side), restaurants)
    cfg = dict(base)
    image = os.path.relpath("data/equipetrol.png", os.path.dirname(out) or ".")
    cfg["map"] = dict(base["map"], image=image.replace(os.sep, "/"))
    cfg["nodes"], cfg["streets"] = nodes, streets
    cfg["restaurants"] = [{"id": f"r{i}", "name": f"R{i}", "node": f"g{cell // side}_{cell % side}",
                           "pickupSlots": 2, "prepTimeMs": [60000, 600000]} for i, cell in enumerate(cells)]
    center = f"g{side // 2}_{side // 2}"
    cfg["fleet"] = dict(base["fleet"], couriers=couriers, startNode=center)
    cfg["orders"] = dict(base["orders"], meanIntervalMs=interval, maxPending=5000)
    cfg["incidents"] = {"breakdownProbability": 0.0}
    cfg["simulation"] = {"durationS": duration, "timeScale": base["simulation"]["timeScale"]}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    print(f"{out}: {len(nodes)} nodes, {len(streets)} streets, {restaurants} restaurants, {couriers} couriers")


if __name__ == "__main__":
    main()