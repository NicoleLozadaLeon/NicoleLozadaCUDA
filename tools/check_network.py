#!/usr/bin/env python3
"""Checks a traced network: counts, endpoints, edge lengths, connectivity with one-way rules, spread."""
import json, math, sys

W, S, E, N = -63.205248240852164, -17.77504707884483, -63.187351138452286, -17.750567044823004
PATH = sys.argv[1] if len(sys.argv) > 1 else "tools/network_draft.json"

def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))

def reach(start, adj):
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen

def main():
    with open(PATH, encoding="utf-8") as f:
        d = json.load(f)
    nodes = {n["id"]: n for n in d["nodes"]}
    streets = d["streets"]
    restaurants = d.get("restaurants", [])
    problems = []
    n_one = sum(1 for s in streets if s["oneWay"])
    print(f"{len(nodes)} nodes, {len(streets)} streets, {n_one} one-way, {len(restaurants)} restaurants")
    if len(nodes) < 25 or len(streets) < 40 or len(restaurants) < 6:
        problems.append("below the brief minimum (25 nodes, 40 streets, 6 restaurants)")

    adj = {i: set() for i in nodes}
    radj = {i: set() for i in nodes}
    seen_pairs, lengths = set(), []
    for s in streets:
        a, b = s["from"], s["to"]
        if a not in nodes or b not in nodes:
            problems.append(f"street {s['id']} uses an unknown node")
            continue
        if a == b:
            problems.append(f"street {s['id']} connects a node to itself")
        key = (a, b) if s["oneWay"] else tuple(sorted((a, b)))
        if key in seen_pairs:
            problems.append(f"duplicate street {a}-{b}")
        seen_pairs.add(key)
        adj[a].add(b); radj[b].add(a)
        if not s["oneWay"]:
            adj[b].add(a); radj[a].add(b)
        m = haversine_m(nodes[a]["lat"], nodes[a]["lon"], nodes[b]["lat"], nodes[b]["lon"])
        lengths.append((m, s["id"], a, b))

    lengths.sort()
    print(f"street length: min {lengths[0][0]:.0f} m, max {lengths[-1][0]:.0f} m, mean {sum(x[0] for x in lengths)/len(lengths):.0f} m")
    for m, sid, a, b in lengths:
        if m > 350 or m < 15:
            print(f"  check {sid} ({a}-{b}): {m:.0f} m")

    isolated = [i for i in nodes if not adj[i] and not radj[i]]
    if isolated:
        problems.append(f"isolated nodes: {isolated}")
    start = next(iter(nodes))
    fwd, back = reach(start, adj), reach(start, radj)
    if len(fwd) != len(nodes):
        problems.append(f"not reachable from {start}: {sorted(set(nodes) - fwd)}")
    if len(back) != len(nodes):
        problems.append(f"cannot reach {start}: {sorted(set(nodes) - back)}")

    grid = [[0] * 3 for _ in range(6)]
    for n in nodes.values():
        c = min(2, max(0, int((n["lon"] - W) / (E - W) * 3)))
        r = min(5, max(0, int((N - n["lat"]) / (N - S) * 6)))
        grid[r][c] += 1
    print("nodes per cell (6 rows, top to bottom, x 3 columns):")
    for row in grid:
        print("  ", row)
    empty_rows = [i + 1 for i, row in enumerate(grid) if sum(row) == 0]
    if empty_rows:
        problems.append(f"no nodes in rows {empty_rows}: the network does not run along the whole length of the chosen side")

    used = [r["node"] for r in restaurants]
    for r in restaurants:
        if r["node"] not in nodes:
            problems.append(f"restaurant {r['name']} points to an unknown node")
    if len(set(used)) != len(used):
        print("note: two restaurants share a node")

    print("PROBLEMS:" if problems else "OK: all checks passed")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
