#!/usr/bin/env python3
"""Summarise the runs of tools/run_experiments.sh as a markdown table (mean of the runs of each case).

usage (from the repository root):  python3 tools/summarize_runs.py > experiments/results.md
"""
import glob
import json
import os
import re
import statistics


def natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def main():
    cases = {}
    for summary in glob.glob("runs/*_r[0-9]*.summary"):
        base = summary[: -len(".summary")]
        case = re.sub(r"_r\d+$", "", os.path.basename(base))
        row = {}
        try:
            with open(summary, encoding="utf-8") as f:
                row.update(json.loads(f.read().strip().splitlines()[-1]))
            with open(base + ".err", encoding="utf-8") as f:
                err = f.read()
        except (OSError, ValueError):
            continue
        m = re.search(r"mode=(\w+) K=(\d+) rankings=(\d+) estimates=(\d+) late=(\d+) failed=(\d+) avgRankUs=(\d+) maxRankUs=(\d+)", err)
        if m:
            for key, value in zip(["mode", "K", "rankings", "estimates", "late", "failed", "avgRankUs", "maxRankUs"], m.groups()):
                row[key] = value if key == "mode" else int(value)
        g = re.search(r"shortest paths in ([\d.]+) ms with (\d+) tasks \(grain (\d+)\)", err)
        if g:
            row["graphMs"], row["tasks"], row["grain"] = float(g.group(1)), int(g.group(2)), int(g.group(3))
        cases.setdefault(case, []).append(row)

    def mean(rows, key, scale=1.0):
        values = [r[key] for r in rows if key in r]
        return f"{statistics.mean(values) / scale:.1f}" if values else "-"

    print("| case | runs | avgRankUs | maxRankUs | late | delivered | rejected | pending | avgDelivery min | maxDelivery min | graph ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in sorted(cases, key=natural):
        rows = cases[case]
        print(f"| {case} | {len(rows)} | {mean(rows, 'avgRankUs')} | {mean(rows, 'maxRankUs')} | {mean(rows, 'late')} | "
              f"{mean(rows, 'delivered')} | {mean(rows, 'rejected')} | {mean(rows, 'pending')} | "
              f"{mean(rows, 'avgDeliveryMs', 60000)} | {mean(rows, 'maxDeliveryMs', 60000)} | {mean(rows, 'graphMs')} |")


if __name__ == "__main__":
    main()
