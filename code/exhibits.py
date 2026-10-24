import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

for path in ["/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf",
             "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf",
             "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf"]:
    try:
        font_manager.fontManager.addfont(path)
    except (FileNotFoundError, RuntimeError):
        pass

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLACK = "#000000"
DARK = "#4a4a4a"
MID = "#8c8c8c"
LIGHT = "#d4d4d4"
CELLS = ["Stable part-time", "Fixed-term", "Intermittent"]
WIDTH = 6.3


def panel_title(ax, letter, text):
    ax.set_title(f"Panel {letter}. {text}", style="italic")


def exhibit_1(pilot, evaluation):
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 1.85), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.08}, constrained_layout=True)
    rows = pilot["groups"] + [pilot["overall"]]
    y = np.arange(4)[::-1]
    effect = np.array([r["effect"] for r in rows]) / 1000
    half = np.array([1.96 * r["se"] for r in rows]) / 1000
    ax[0].errorbar(effect[:3], y[:3], xerr=half[:3], fmt="o", ms=4, color=BLACK, ecolor=BLACK, elinewidth=0.8, capsize=2.5)
    ax[0].errorbar(effect[3:], y[3:], xerr=half[3:], fmt="D", ms=4, color=BLACK, mfc="white", ecolor=BLACK, elinewidth=0.8, capsize=2.5)
    ax[0].axvline(0, color=MID, lw=0.6, ls=(0, (3, 2)))
    ax[0].axhline(0.5, color=LIGHT, lw=0.6)
    ax[0].set_yticks(y, CELLS + ["All workers"])
    ax[0].set_xlim(-0.1, 2.6)
    ax[0].set_ylim(-0.5, 3.5)
    ax[0].set_xlabel("Earnings gain over 24 months\n(thousands of euros)")
    panel_title(ax[0], "A", "Average effect of the offer")
    shares = evaluation["continuous"]["allowance"]
    top = np.r_[shares["unrestricted"]["cells"], shares["unrestricted"]["weighted"]] * 100
    independent = np.r_[shares["independent"]["cells"], shares["independent"]["weighted"]] * 100
    persistent = np.r_[shares["persistent"]["cells"], shares["persistent"]["weighted"]] * 100
    yc = np.arange(4)[::-1]
    for k, row in enumerate(yc):
        ax[1].barh(row, top[k], height=0.36, color=LIGHT, edgecolor="none", zorder=1)
        ax[1].plot([top[k], top[k]], [row - 0.2, row + 0.2], color=BLACK, lw=0.9, zorder=2)
        ax[1].text(top[k] + 1.5, row, f"{top[k]:.0f}%", va="center", fontsize=8)
    ax[1].scatter(independent, yc, marker="o", s=22, color=BLACK, zorder=3)
    ax[1].scatter(persistent, yc, marker="o", s=22, facecolor="white", edgecolor=BLACK, linewidth=0.8, zorder=3)
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(-0.5, 3.5)
    ax[1].set_yticks([])
    ax[1].axhline(0.5, color=LIGHT, lw=0.6)
    ax[1].set_xlabel("Workers whose disposable income falls\n(percent)")
    panel_title(ax[1], "B", "Share of losers consistent with the pilot")
    handles = [Patch(color=LIGHT, label="Range consistent with the pilot"),
               Line2D([], [], ls="", marker="o", ms=4.5, color=BLACK, label="Course reshuffles the ranking"),
               Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=BLACK, label="Course keeps the ranking")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, handlelength=1.4, columnspacing=1.4)
    fig.savefig("../paper/figures/exhibit_1.pdf")
    fig.savefig("../paper/figures/exhibit_1.png", dpi=200)
    plt.close(fig)


def exhibit_2(evaluation):
    scale = 100.0 / evaluation["reference_distance"]["independent_to_persistent"]
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.05), gridspec_kw={"width_ratios": [1.25, 0.85]}, constrained_layout=True)
    styles = {"independent": dict(color=BLACK, ls="-", marker="o", ms=3.2, mfc=BLACK),
              "persistent": dict(color=BLACK, ls=(0, (4, 2)), marker="o", ms=3.2, mfc="white")}
    for ref in ["independent", "persistent"]:
        curve = evaluation["grid_curves"][ref]
        ax[0].plot([scale * p["radius"] for p in curve], [100 * p["aggregate"] for p in curve], lw=0.9, **styles[ref])
    ax[0].axhline(100 * evaluation["grid_unrestricted"]["weighted"], color=MID, lw=0.8, ls=(0, (1, 1.5)))
    ax[0].axhline(10, color=MID, lw=0.6)
    ax[0].text(0.62 * 0.82 * scale, 11.5, "At most 10% lose", fontsize=7.5, color=DARK)
    breakdown = scale * evaluation["breakdown_persistent_10pct"]
    ax[0].plot([breakdown, breakdown], [0, 10], color=DARK, lw=0.6, ls=(0, (2, 2)))
    ax[0].annotate(f"{breakdown:.1f}%", (breakdown, 0), xytext=(breakdown + 1.0, 2.0), fontsize=7.5, color=DARK)
    ax[0].set_xlim(0, 0.82 * scale)
    ax[0].set_ylim(0, 80)
    ax[0].set_xlabel("Divergence from the scenario\n(percent of the gap between the scenarios)")
    ax[0].set_ylabel("Percent of workers")
    ax[0].set_yticks([0, 20, 40, 60, 80])
    panel_title(ax[0], "A", "Worst-case share of losers, all workers")
    x = np.arange(3)
    width = 0.36
    fills = {"independent": DARK, "persistent": "white"}
    for i, ref in enumerate(["independent", "persistent"]):
        cells = [p for p in evaluation["grid_curves"][ref] if p["radius"] == 0.1][0]["cells"]
        ax[1].bar(x + (i - 0.5) * width, [100 * v for v in cells], width, color=fills[ref], edgecolor=BLACK, linewidth=0.6)
    ax[1].axhline(10, color=MID, lw=0.6)
    ax[1].set_xticks(x, ["Stable\npart-time", "Fixed-term", "Inter-\nmittent"])
    ax[1].tick_params(axis="x", length=0)
    ax[1].set_ylim(0, 55)
    ax[1].set_ylabel("Percent of workers")
    ax[1].set_yticks([0, 10, 20, 30, 40, 50])
    panel_title(ax[1], "B", "By cell, at 5% of the gap")
    handles = [Line2D([], [], lw=0.9, label="Independent scenario", **styles["independent"]),
               Line2D([], [], lw=0.9, label="Persistent scenario", **styles["persistent"]),
               Line2D([], [], color=MID, lw=0.8, ls=(0, (1, 1.5)), label="No restriction on the matching")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, handlelength=2.2, columnspacing=1.6)
    fig.savefig("../paper/figures/exhibit_2_losers.pdf")
    fig.savefig("../paper/figures/exhibit_2_losers.png", dpi=200)
    plt.close(fig)


def main():
    pilot = json.load(open("../results/pilot.json"))
    evaluation = json.load(open("../results/evaluation.json"))
    exhibit_1(pilot, evaluation)
    exhibit_2(evaluation)


if __name__ == "__main__":
    main()
