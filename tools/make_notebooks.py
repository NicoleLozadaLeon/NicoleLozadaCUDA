#!/usr/bin/env python3
"""Write the EC2 report notebook: analysis/report.ipynb (tables and plots from the CSV files in results/).

usage: python3 tools/make_notebooks.py
"""
import json
import os

REPORT = [
    ("md", "# EC2 report: tables and plots\nEvery number in the report comes from the CSV files in `results/` "
           "(written by `delivery_sim --bench`). CPU and GPU were measured on the same laptop."),
    ("code", "import glob\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
             "CPU_MACHINE = 'laptop'     # the machine whose CPU columns go into the verdict\n"
             "GPU_MACHINE = 'laptop'     # the machine whose GPU columns go into the verdict\n"
             "df = pd.concat([pd.read_csv(f) for f in glob.glob('../results/bench_*.csv')], ignore_index=True)\n"
             "print(df.machine.unique(), len(df), 'rows')\n"
             "assert (df.correct == 'yes').all(), df[df.correct != 'yes']\n"
             "APPS = list(dict.fromkeys(df.app))\n"
             "med = df.groupby(['machine', 'app', 'concept', 'size', 'size_label', 'version', 'threads']).ms.agg(\n"
             "    median='median', lo='min', hi='max').reset_index()\nmed.head()"),
    ("md", "## Table per application (CPU and GPU on the laptop)"),
    ("code", "def table(app, cpu_machine=CPU_MACHINE, gpu_machine=GPU_MACHINE):\n"
             "    m = med[med.app == app]\n"
             "    cpu, gpu = m[m.machine == cpu_machine], m[m.machine == gpu_machine]\n"
             "    rows = []\n"
             "    for size, label in m[['size', 'size_label']].drop_duplicates().sort_values('size').values:\n"
             "        c, g = cpu[cpu['size'] == size], gpu[gpu['size'] == size]\n"
             "        one = c[c.version == 'cpu1']['median'].min()\n"
             "        multi = c[c.version == 'cpuN'].sort_values('median').head(1)\n"
             "        best = min(one, multi['median'].min()) if len(multi) else one\n"
             "        k = g[g.version == 'gpu_kernel']['median'].min()\n"
             "        e = g[g.version == 'gpu_e2e']['median'].min()\n"
             "        rows.append({'size': label, 'CPU 1 (ms)': one,\n"
             "                     f\"CPU N={int(multi.threads.iloc[0]) if len(multi) else '-'} (ms)\": multi['median'].min(),\n"
             "                     'GPU kernel (ms)': k, 'GPU e2e (ms)': e,\n"
             "                     'speedup kernel': best / k, 'speedup e2e': best / e, 'correct': 'yes'})\n"
             "    return pd.DataFrame(rows)\n\n"
             "for app in APPS:\n    print(app)\n    display(table(app).round(3))"),
    ("md", "## Plot 1: time vs size (log-log) with the crossover"),
    ("code", "def crossover(app):\n"
             "    t = table(app)\n"
             "    best = t[[c for c in t.columns if c.startswith('CPU')]].min(axis=1)\n"
             "    wins = t[t['GPU e2e (ms)'] < best]\n"
             "    return wins['size'].iloc[0] if len(wins) else 'never'\n\n"
             "fig, axes = plt.subplots(2, (len(APPS) + 1) // 2, figsize=(4 * ((len(APPS) + 1) // 2), 7))\n"
             "for ax, app in zip(axes.flat, APPS):\n"
             "    m = med[med.app == app]\n"
             "    for machine, version, style in [(CPU_MACHINE, 'cpu1', 'o-'), (CPU_MACHINE, 'cpuN', 's-'),\n"
             "                                    (GPU_MACHINE, 'gpu_kernel', '^--'), (GPU_MACHINE, 'gpu_e2e', 'v-')]:\n"
             "        s = m[(m.machine == machine) & (m.version == version)]\n"
             "        if version == 'cpuN' and len(s):\n"
             "            s = s[s.threads == s.threads.max()]\n"
             "        ax.loglog(s['size'], s['median'], style, label=f'{version} ({machine})')\n"
             "    ax.set_title(f'{app}: crossover {crossover(app)}', fontsize=9)\n"
             "    ax.set_xlabel('size'); ax.set_ylabel('ms')\n"
             "axes.flat[0].legend(fontsize=7)\nplt.tight_layout(); plt.savefig('../results/time_vs_size.png', dpi=150)"),
    ("md", "## Plot 2: share of the GPU time spent in copies vs kernels"),
    ("code", "fig, axes = plt.subplots(2, (len(APPS) + 1) // 2, figsize=(4 * ((len(APPS) + 1) // 2), 7))\n"
             "for ax, app in zip(axes.flat, APPS):\n"
             "    m = med[(med.app == app) & (med.machine == GPU_MACHINE)]\n"
             "    k = m[m.version == 'gpu_kernel'].set_index('size_label')['median']\n"
             "    e = m[m.version == 'gpu_e2e'].set_index('size_label')['median']\n"
             "    order = m[m.version == 'gpu_e2e'].sort_values('size').size_label\n"
             "    share = pd.DataFrame({'kernel': (k / e)[order], 'copies + alloc': (1 - k / e)[order]})\n"
             "    share.plot.bar(stacked=True, ax=ax, legend=False, fontsize=7)\n"
             "    ax.set_title(app, fontsize=9); ax.set_ylim(0, 1)\n"
             "axes.flat[0].legend(fontsize=7)\nplt.tight_layout(); plt.savefig('../results/copy_share.png', dpi=150)"),
    ("md", "## Thread-count sweep (laptop)\nRun the laptop bench with `--threads 1,2,4,8,14,20` to fill this plot."),
    ("code", "s = med[(med.machine == CPU_MACHINE) & (med.version == 'cpuN')]\n"
             "largest = s.groupby('app')['size'].transform('max')\ns = s[s['size'] == largest]\n"
             "for app, g in s.groupby('app'):\n"
             "    one = med[(med.machine == CPU_MACHINE) & (med.app == app) & (med.version == 'cpu1') &\n"
             "              (med['size'] == g['size'].iloc[0])]['median'].iloc[0]\n"
             "    g = g.sort_values('threads')\n"
             "    plt.plot(g.threads, one / g['median'], 'o-', label=app)\n"
             "plt.xlabel('threads'); plt.ylabel('speedup vs CPU 1'); plt.legend(fontsize=7)\n"
             "plt.savefig('../results/thread_sweep.png', dpi=150)"),
    ("md", "## AoS vs SoA (application eta): effective bandwidth\nUseful bytes per order x courier pair: "
           "freeNode 4 + freeAtMs 8 + one table entry 4 + the output 4 = 20 bytes."),
    ("code", "m = med[(med.app == 'eta') & (med.machine == GPU_MACHINE) & med.version.isin(['gpu_kernel', 'gpu_aos_kernel'])]\n"
             "bw = m.assign(GBps=20 * m['size'] / (m['median'] / 1000) / 1e9)\n"
             "bw.pivot(index='size_label', columns='version', values='GBps').round(1)"),
    ("md", "## AoS vs SoA (application foodwait): effective bandwidth\nUseful bytes per order in a bag: readyAtMs 8 + pickedUp 1 + the output 4 = 13 bytes."),
    ("code", "m = med[(med.app == 'foodwait') & (med.machine == GPU_MACHINE) & med.version.isin(['gpu_kernel', 'gpu_aos_kernel'])]\n"
             "bw = m.assign(GBps=13 * m['size'] / (m['median'] / 1000) / 1e9)\n"
             "bw.pivot(index='size_label', columns='version', values='GBps').round(1)"),
    ("md", "## Roofline (RTX 3060 Laptop: 3840 CUDA cores x 2 FLOP x 2.1 GHz = 16.1 TFLOP/s FP32; 192-bit GDDR6 at 14 Gbps = 336 GB/s)\nFLOP and byte counts per kernel, derived from the code "
           "(write the derivation in the report). Achieved performance uses the kernel time at the largest size."),
    ("code", "PEAK_FLOPS, PEAK_BW = 2 * 3840 * 2.1e9, 336e9\n"
             "def counts(app, size):\n"
             "    n = size\n"
             "    if app == 'fw':\n"
             "        N = int(np.ceil(n / 32) * 32)\n"
             "        return 2 * N ** 3, (N // 32) * N * N * 4 * 4\n"
             "    if app == 'eta': return 12 * n, 20 * n\n"
             "    if app == 'argmin': return 2 * n, 4 * n\n"
             "    if app == 'hist': return 3 * n, 4 * n\n"
             "    if app == 'group': return 2 * 64 * n, 12 * n\n"
             "    if app == 'conv': return 2 * 25 * n, 8 * n\n"
             "    if app == 'stencil': return 100 * 6 * n, 100 * 12 * n\n"
             "    if app == 'near': return 7 * n, 8 * n / 128\n"
             "    if app == 'foodwait': return 4 * n, 13 * n\n"
             "    return np.nan, np.nan\n"
             "pts = []\n"
             "for app in APPS:\n"
             "    m = med[(med.app == app) & (med.machine == GPU_MACHINE) & (med.version == 'gpu_kernel')]\n"
             "    if not len(m): continue\n"
             "    r = m.sort_values('size').iloc[-1]\n"
             "    flops, bytes_ = counts(app, r['size'])\n"
             "    pts.append((app, flops / bytes_, flops / (r['median'] / 1000)))\n"
             "ai = np.logspace(-2, 3, 100)\n"
             "plt.loglog(ai, np.minimum(PEAK_FLOPS, ai * PEAK_BW), 'k-')\n"
             "for app, x, y in pts:\n    plt.loglog(x, y, 'o'); plt.annotate(app, (x, y), fontsize=8)\n"
             "plt.xlabel('arithmetic intensity (FLOP/byte)'); plt.ylabel('FLOP/s')\n"
             "plt.savefig('../results/roofline.png', dpi=150)\npd.DataFrame(pts, columns=['app', 'FLOP/byte', 'FLOP/s'])"),
    ("md", "## Effect on the whole simulation"),
    ("code", "sims = pd.concat([pd.read_csv(f) for f in glob.glob('../results/sim_*.csv')], ignore_index=True)\nsims"),
]


def notebook(cells):
    out = []
    for kind, text in cells:
        lines = text.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    os.makedirs("analysis", exist_ok=True)
    for name, cells in [("report", REPORT)]:
        with open(f"analysis/{name}.ipynb", "w", encoding="utf-8") as f:
            json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
        print(f"analysis/{name}.ipynb: {len(cells)} cells")


if __name__ == "__main__":
    main()