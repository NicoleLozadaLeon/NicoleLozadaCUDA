#!/usr/bin/env python3
"""Run the simulation with and without the GPU and write one CSV row per run (EC2 report, section 5).

usage: python3 tools/sim_compare.py <binary> <config.json> <machine> <out.csv> [--window <seconds>]
Headless by default (it runs until durationS). With --window the window is opened and the run is stopped
with SIGINT after <seconds>, so the frame rate is measured too.
"""
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time


def run(binary, config, gpu, window_s):
    env = dict(os.environ, EQ_GPU="1" if gpu else "0")
    cmd = [binary, config, "--log", f"/tmp/sim_compare_{int(gpu)}.log"]
    if window_s is None:
        cmd.append("--headless")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if window_s is not None:
        time.sleep(window_s)
        proc.send_signal(signal.SIGINT)
    out, err = proc.communicate()
    wall = time.time() - t0
    lines = [l for l in out.splitlines() if l.startswith("{")]
    stats, summary = json.loads(lines[-2]), json.loads(lines[-1])
    disp = dict(re.findall(r"(\w+)=([\w.]+)", next(l for l in err.splitlines() if l.startswith("dispatcher:"))))
    fps = re.search(r"fps=([\d.]+)", err)
    graph = re.search(r"shortest paths in ([\d.]+) ms", err)
    return {
        "mode": "gpu" if gpu else "cpu",
        "backend": stats["backend"],
        "graph_ms": graph.group(1) if graph else "",
        "rankings": disp.get("rankings"),
        "avg_rank_us": disp.get("avgRankUs"),
        "max_rank_us": disp.get("maxRankUs"),
        "late_estimates": disp.get("late"),
        "created": summary["created"],
        "delivered": summary["delivered"],
        "rejected": summary["rejected"],
        "avg_delivery_ms": summary["avgDeliveryMs"],
        "stats_ms": round(stats["statsMs"], 3),
        "fps": fps.group(1) if fps else "",
        "wall_s": round(wall, 1),
    }


def main():
    args = sys.argv[1:]
    window_s = None
    if "--window" in args:
        i = args.index("--window")
        window_s = float(args[i + 1])
        del args[i:i + 2]
    if len(args) != 4:
        sys.exit(__doc__)
    binary, config, machine, out = args
    rows = []
    for gpu in (False, True):
        row = {"machine": machine, "config": os.path.basename(config), **run(binary, config, gpu, window_s)}
        print(row)
        rows.append(row)
    new = not os.path.exists(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
