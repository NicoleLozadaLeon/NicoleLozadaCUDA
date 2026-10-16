#!/usr/bin/env python3
"""Write a synthetic N x N grid configuration, for the graph experiment (E4) only.

usage: python3 tools/make_grid_config.py <n> <out.json>
The grid is spread over the bounds of config/equipetrol.json, every street is two-way,
6 restaurants sit on the grid, and the run lasts 1 simulated second (only the start-up
time of the shortest-path tables matters for this experiment).
"""
import json
import os
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n, out = int(sys.argv[1]), sys.argv[2]
    with open("config/equipetrol.json", encoding="utf-8") as f:
        base = json.load(f)
    b = base["map"]["bounds"]
    nodes, streets = [], []
    for r in range(n):
        for c in range(n):
            lat = b["north"] + (b["south"] - b["north"]) * (r + 0.5) / n
            lon = b["west"] + (b["east"] - b["west"]) * (c + 0.5) / n
            nodes.append({"id": f"g{r}_{c}", "lat": lat, "lon": lon})
            if c + 1 < n:
                streets.append({"id": f"h{r}_{c}", "from": f"g{r}_{c}", "to": f"g{r}_{c + 1}", "oneWay": False})
            if r + 1 < n:
                streets.append({"id": f"v{r}_{c}", "from": f"g{r}_{c}", "to": f"g{r + 1}_{c}", "oneWay": False})
    cfg = dict(base)
    image = os.path.relpath("data/equipetrol.png", os.path.dirname(out) or ".")   # relative to the output file
    cfg["map"] = dict(base["map"], image=image.replace(os.sep, "/"))
    cfg["nodes"], cfg["streets"] = nodes, streets
    cfg["restaurants"] = [dict(r, node=f"g{(i * 7) % n}_{(i * 13) % n}") for i, r in enumerate(base["restaurants"])]
    cfg["fleet"] = dict(base["fleet"], startNode="g0_0")
    cfg["simulation"] = {"durationS": 1, "timeScale": 60}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    print(f"{out}: {len(nodes)} nodes, {len(streets)} streets")


if __name__ == "__main__":
    main()
