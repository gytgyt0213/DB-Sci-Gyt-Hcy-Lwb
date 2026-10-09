"""Render the reproduction result as a figure for the report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "spider", "resdsql")

# Values read from work/logs/eval_baseline_top1.log and eval_cyclesql.log
blocks = ["easy", "medium", "hard", "extra", "all"]
baseline = [0.812, 0.067, 0.357, 0.000, 0.323]
cyclesql = [0.875, 0.067, 0.500, 0.000, 0.371]
counts = [16, 30, 14, 2, 62]

x = np.arange(len(blocks))
w = 0.38

fig, ax = plt.subplots(figsize=(8.2, 4.6))
b1 = ax.bar(x - w / 2, baseline, w, label="RESDSQL top-1 (baseline)", color="#8c9bab")
b2 = ax.bar(x + w / 2, cyclesql, w, label="CycleSQL loop (reproduced)", color="#2f6fb0")

for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=8.5)

ax.set_ylabel("Execution accuracy")
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n(n={c})" for b, c in zip(blocks, counts)])
ax.set_ylim(0, 1.05)
ax.set_title("CycleSQL reproduction on Spider dev / wta_1  (62 questions, RESDSQL beam=8)\n"
             "32.3% -> 37.1% execution accuracy (+4.8 pts, +14.9% relative)",
             fontsize=10.5)
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
path = os.path.join(OUT, "fig_accuracy.png")
fig.savefig(path, dpi=200)
print("saved", path)

with open(os.path.join(OUT, "result_table.json"), "w", encoding="utf-8") as f:
    json.dump({"difficulty": blocks, "counts": counts,
               "baseline_top1": baseline, "cyclesql": cyclesql}, f, indent=2)
