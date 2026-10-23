import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

evaluation = json.load(open("../results/evaluation.json"))
scale = 100.0 / evaluation["reference_distance"]["independent_to_persistent"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "text.color": "#17334d", "axes.labelcolor": "#17334d"})
colors = {"independent": "#087f8c", "persistent": "#b35b32"}
labels = {"independent": "Independent reference", "persistent": "Persistent reference, κ = 0.10"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1.2, 0.8]}, constrained_layout=True)
for ref in ["independent", "persistent"]:
    curve = evaluation["grid_curves"][ref]
    ax[0].plot([scale * p["radius"] for p in curve], [100 * p["aggregate"] for p in curve], "-o", ms=3.5, color=colors[ref], label=labels[ref])
ax[0].axhline(100 * evaluation["grid_unrestricted"]["weighted"], color="#17334d", ls=":", label="Worst case without restrictions")
ax[0].axhline(10, color="#777", ls="--", lw=1, label="Claim: at most 10% lose")
breakdown = scale * evaluation["breakdown_persistent_10pct"]
ax[0].scatter([breakdown], [10], color=colors["persistent"], zorder=5)
ax[0].annotate(f"claim breaks at {breakdown:.1f}%", (breakdown, 10), xytext=(breakdown + 1.5, 4), fontsize=8, color=colors["persistent"])
ax[0].set(xlabel="Distance from the reference, % of the distance between the two references", ylabel="Worst-case share of losers, %", title="A. Losers after the allowance as the distance grows", xlim=(0, 0.82 * scale), ylim=(0, 82))
ax[0].legend(frameon=False, fontsize=7.5, loc="center right")
x = np.arange(3)
width = 0.36
for i, ref in enumerate(["independent", "persistent"]):
    cells = [p for p in evaluation["grid_curves"][ref] if p["radius"] == 0.1][0]["cells"]
    ax[1].bar(x + (i - 0.5) * width, [100 * v for v in cells], width, color=colors[ref], label=labels[ref])
ax[1].axhline(10, color="#777", ls="--", lw=1)
ax[1].set_xticks(x, ["Stable\npart-time", "Fixed-term", "Intermittent"])
ax[1].set(ylabel="Worst-case share of losers, %", title="B. Incidence by cell at 5% distance", ylim=(0, 58))
ax[1].legend(frameon=False, fontsize=7.5)
for ext in ["pdf", "png"]:
    fig.savefig(f"../paper/figures/exhibit_2_losers.{ext}", dpi=160)
