"""
ContextStress Paper — Publication-Quality Figure Generation (v3)

Generates five publication-quality figures from real experimental data.
Run: python3 generate_figures_v3.py

Data sources: results/snr100_v2/, results/snr50_v2/, results/snr25/
Outputs (PDF + PNG for each):
  figures_v3/fig1_hero_collapse.*           — Hero collapse curve (SNR=50%)
  figures_v3/fig2_sigmoid_vs_linear.*       — Sigmoid vs linear fit for T3 SNR=50%
  figures_v3/fig3_snr_effect.*              — SNR effect: 3 subplots
  figures_v3/fig4_calibration_scatter.*     — Calibration scatter
  figures_v3/fig5_t2_snr_comparison.*       — T2 collapse across SNR levels
"""

import csv
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

# ── Paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

DATA_FILES = [
    os.path.join(BASE, "results/snr100_v2/gpt-4o-mini_20260401_005453/results_corrected.csv"),
    os.path.join(BASE, "results/snr50_v2/gpt-4o-mini_20260401_203234/results_corrected.csv"),
    os.path.join(BASE, "results/snr25/gpt-4o-mini_20260331_222409/results_corrected.csv"),
]

# ── Global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# ── Color palette ────────────────────────────────────────────────────────
PAL = {
    "blue":   "#2196F3",
    "orange": "#FF9800",
    "red":    "#F44336",
    "purple": "#9C27B0",
    "gray":   "#757575",
}
FAMILY_COLORS = {"T1": PAL["blue"], "T2": PAL["orange"], "T3": PAL["red"], "T4": PAL["purple"]}
FAMILY_MARKERS = {"T1": "o", "T2": "D", "T3": "s", "T4": "^"}
FAMILY_LABELS = {
    "T1": "T1 (single-hop)",
    "T2": "T2 (two-hop)",
    "T3": "T3 (three-hop)",
    "T4": "T4 (four-hop)",
}

SNR_LABELS = {1.0: "SNR=100%", 0.5: "SNR=50%", 0.25: "SNR=25%"}
SNR_NICE = {1.0: "100%", 0.5: "50%", 0.25: "25%"}


# ── Load data ────────────────────────────────────────────────────────────
def load_all():
    rows = []
    for fpath in DATA_FILES:
        with open(fpath) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows.append(r)
    return rows


def group_stats(rows):
    """Group by (task_family, cu_target, snr_target) -> accuracy, SE, n, mean_confidence."""
    groups = {}
    for r in rows:
        key = (r["task_family"], float(r["cu_target"]), float(r["snr_target"]))
        if key not in groups:
            groups[key] = {"correct": [], "confidence": []}
        groups[key]["correct"].append(r["correct"] == "True")
        conf_str = r["confidence"]
        if conf_str:
            groups[key]["confidence"].append(float(conf_str))

    stats = {}
    for key, vals in groups.items():
        n = len(vals["correct"])
        p = sum(vals["correct"]) / n
        se = math.sqrt(p * (1 - p) / n) if n > 0 else 0
        mean_conf = np.mean(vals["confidence"]) if vals["confidence"] else 0
        stats[key] = {"acc": p * 100, "se": se * 100, "n": n, "mean_conf": mean_conf}
    return stats


print("Loading data...")
all_rows = load_all()
stats = group_stats(all_rows)
print(f"  Loaded {len(all_rows)} rows, {len(stats)} conditions")

# Build handy arrays
CU_VALS = sorted(set(k[1] for k in stats))  # [0.05, 0.15, 0.3, 0.5, 0.75, 0.95]
CU_PCT = [cu * 100 for cu in CU_VALS]       # [5, 15, 30, 50, 75, 95]
SNR_VALS = sorted(set(k[2] for k in stats)) # [0.25, 0.5, 1.0]
FAMILIES = ["T1", "T2", "T3", "T4"]


def get_acc_se(family, snr):
    """Return (cu_pct_array, acc_array, se_array) for a family at given SNR."""
    cu_list, acc_list, se_list = [], [], []
    for cu in CU_VALS:
        key = (family, cu, snr)
        if key in stats:
            cu_list.append(cu * 100)
            acc_list.append(stats[key]["acc"])
            se_list.append(stats[key]["se"])
    return np.array(cu_list), np.array(acc_list), np.array(se_list)


def sigmoid(x, L, U, k, x0):
    """Four-parameter logistic sigmoid."""
    return L + (U - L) / (1.0 + np.exp(k * (x - x0)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Hero Collapse Curve — SNR=50%, all 4 families
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Hero Collapse Curve (SNR=50%)")

fig, ax = plt.subplots(figsize=(7, 4.5))
CU_smooth = np.linspace(2, 98, 300)

for fam in FAMILIES:
    cu, acc, se = get_acc_se(fam, 0.5)
    color = FAMILY_COLORS[fam]
    marker = FAMILY_MARKERS[fam]

    # Try sigmoid fit for smooth line
    try:
        p0 = [20, 100, 0.04, 50]
        bounds = ([0, 50, 0.001, 1], [80, 105, 1.0, 99])
        popt, _ = curve_fit(sigmoid, cu, acc, p0=p0, bounds=bounds, maxfev=10000)
        y_smooth = sigmoid(CU_smooth, *popt)
        se_interp = np.interp(CU_smooth, cu, se)
        ax.fill_between(CU_smooth, y_smooth - se_interp, y_smooth + se_interp,
                        color=color, alpha=0.10, zorder=1)
        ax.plot(CU_smooth, y_smooth, color=color, linewidth=2, zorder=2)
    except Exception:
        ax.plot(cu, acc, color=color, linewidth=2, zorder=2)

    # Data points with error bars
    ax.errorbar(cu, acc, yerr=se, fmt="none", ecolor=color,
                capsize=3, capthick=1, elinewidth=1, zorder=3, alpha=0.7)
    ax.scatter(cu, acc, color=color, marker=marker, s=55, zorder=4,
               edgecolors="white", linewidths=0.6, label=FAMILY_LABELS[fam])

# 80% reliability line
ax.axhline(y=80, color=PAL["gray"], linestyle=":", linewidth=1, alpha=0.5)
ax.text(3, 81.5, "80% reliability", fontsize=8, color=PAL["gray"], alpha=0.7)

ax.set_xlabel("Context Utilization (CU, % of window)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Context Collapse Curve — gpt-4o-mini, SNR = 50%")
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.legend(loc="lower left", frameon=True, fancybox=False)
ax.grid(True, axis="both", alpha=0.2)

save(fig, "fig1_hero_collapse")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Sigmoid vs Linear Fit — T3, SNR=50%
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 2: Sigmoid vs Linear Fit (T3, SNR=50%)")

fig, ax = plt.subplots(figsize=(7, 4.5))

cu, acc, se = get_acc_se("T3", 0.5)

# Linear fit
z = np.polyfit(cu, acc, 1)
linear_smooth = np.polyval(z, CU_smooth)
linear_pred = np.polyval(z, cu)
ss_res_lin = np.sum((acc - linear_pred) ** 2)
ss_tot = np.sum((acc - np.mean(acc)) ** 2)
r2_lin = 1 - ss_res_lin / ss_tot

# Sigmoid fit
try:
    p0 = [20, 100, 0.04, 50]
    bounds = ([0, 50, 0.001, 1], [80, 105, 1.0, 99])
    popt, _ = curve_fit(sigmoid, cu, acc, p0=p0, bounds=bounds, maxfev=10000)
    sig_smooth = sigmoid(CU_smooth, *popt)
    sig_pred = sigmoid(cu, *popt)
    ss_res_sig = np.sum((acc - sig_pred) ** 2)
    r2_sig = 1 - ss_res_sig / ss_tot
except Exception:
    # Fallback: just use linear
    popt = None
    r2_sig = 0

# Error band for sigmoid
if popt is not None:
    se_interp = np.interp(CU_smooth, cu, se)
    ax.fill_between(CU_smooth, sig_smooth - se_interp, sig_smooth + se_interp,
                    color=PAL["red"], alpha=0.08, zorder=1)

# Linear fit line
ax.plot(CU_smooth, linear_smooth, "--", color=PAL["gray"], linewidth=2,
        label=f"Linear fit (R² = {r2_lin:.4f})", zorder=2)

# Sigmoid fit line
if popt is not None:
    ax.plot(CU_smooth, sig_smooth, "-", color=PAL["red"], linewidth=2.5,
            label=f"Sigmoid fit (R² = {r2_sig:.4f})", zorder=3)

# Data points with error bars
ax.errorbar(cu, acc, yerr=se, fmt="none", ecolor=PAL["orange"],
            capsize=4, capthick=1.2, elinewidth=1.2, zorder=3)
ax.scatter(cu, acc, s=70, color=PAL["orange"], zorder=5,
           edgecolors="white", linewidths=0.8, label="Observed data")

# Annotate each point with accuracy
for xi, yi in zip(cu, acc):
    ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                xytext=(0, 10), fontsize=8, ha="center", color=PAL["orange"])

ax.axhline(y=80, color=PAL["gray"], linestyle=":", linewidth=1, alpha=0.4)

ax.set_xlabel("Context Utilization (CU, % of window)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Sigmoid vs. Linear Fit — T3, SNR = 50%")
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.legend(loc="lower left", frameon=True, fancybox=False)
ax.grid(True, axis="both", alpha=0.2)

save(fig, "fig2_sigmoid_vs_linear")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: SNR Effect — 3 subplots side by side
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 3: SNR Effect (3 subplots)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

for ax_i, (ax, snr) in enumerate(zip(axes, [0.25, 0.5, 1.0])):
    for fam in FAMILIES:
        cu, acc, se = get_acc_se(fam, snr)
        color = FAMILY_COLORS[fam]
        marker = FAMILY_MARKERS[fam]

        # Smooth line via sigmoid
        try:
            p0 = [20, 100, 0.04, 50]
            bounds = ([0, 50, 0.001, 1], [80, 105, 1.0, 99])
            popt, _ = curve_fit(sigmoid, cu, acc, p0=p0, bounds=bounds, maxfev=10000)
            y_smooth = sigmoid(CU_smooth, *popt)
            ax.plot(CU_smooth, y_smooth, color=color, linewidth=1.8, zorder=2)
        except Exception:
            ax.plot(cu, acc, color=color, linewidth=1.8, zorder=2)

        ax.errorbar(cu, acc, yerr=se, fmt="none", ecolor=color,
                    capsize=2, capthick=0.8, elinewidth=0.8, zorder=3, alpha=0.6)
        label = FAMILY_LABELS[fam] if ax_i == 0 else None
        ax.scatter(cu, acc, color=color, marker=marker, s=40, zorder=4,
                   edgecolors="white", linewidths=0.5, label=label)

    ax.axhline(y=80, color=PAL["gray"], linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("CU (% of window)")
    ax.set_title(f"SNR = {SNR_NICE[snr]}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="both", alpha=0.2)

axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(loc="lower left", frameon=True, fancybox=False, fontsize=8)

fig.suptitle("How Noise Shifts the Context Collapse — gpt-4o-mini", fontsize=14,
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

save(fig, "fig3_snr_effect")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Calibration Scatter
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 4: Calibration Scatter")

fig, ax = plt.subplots(figsize=(7, 4.5))

# Collect per-condition: stated confidence vs actual accuracy, colored by CU
cu_all = []
acc_all = []
conf_all = []
labels_all = []

for (fam, cu, snr), s in stats.items():
    cu_all.append(cu * 100)
    acc_all.append(s["acc"])
    conf_all.append(s["mean_conf"])
    labels_all.append(f"{fam} CU={cu*100:.0f}% SNR={SNR_NICE[snr]}")

cu_all = np.array(cu_all)
acc_all = np.array(acc_all)
conf_all = np.array(conf_all)

# Perfect calibration line
ax.plot([0, 105], [0, 105], "-", color=PAL["gray"], linewidth=1.5, alpha=0.5, zorder=1)
ax.text(55, 48, "Perfect calibration", fontsize=8, color=PAL["gray"],
        rotation=38, ha="center", va="center", alpha=0.6, fontstyle="italic")

# Overconfidence label
ax.text(30, 75, "OVERCONFIDENT", fontsize=11, color=PAL["red"],
        alpha=0.12, fontweight="bold", ha="center", rotation=0)

# Scatter colored by CU
sc = ax.scatter(conf_all, acc_all, c=cu_all, cmap="coolwarm", s=60,
                edgecolors="white", linewidths=0.5, zorder=3, vmin=0, vmax=100)
cbar = fig.colorbar(sc, ax=ax, shrink=0.85, aspect=25, pad=0.02)
cbar.set_label("CU (%)", fontsize=10)

# Annotate SINGLE worst calibration gap (cleanest)
gaps = np.abs(conf_all - acc_all)
worst_idx = np.argmax(gaps)
gap = gaps[worst_idx]
if gap > 50:
    ax.annotate(
        f"Silent failure:\n{gap:.0f}pp gap",
        xy=(conf_all[worst_idx], acc_all[worst_idx]),
        xytext=(55, 20),
        fontsize=9, color=PAL["red"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PAL["red"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=PAL["red"], alpha=0.9),
        zorder=5,
    )

ax.set_xlabel("Stated Confidence (%)")
ax.set_ylabel("Actual Accuracy (%)")
ax.set_title("Calibration Scatter — Confidence vs. Accuracy")
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.15)

save(fig, "fig4_calibration_scatter")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: T2 Collapse Comparison Across SNR Levels
# ══════════════════════════════════════════════════════════════════════════
print("\nFigure 5: T2 Collapse Comparison Across SNR Levels")

fig, ax = plt.subplots(figsize=(7, 4.5))

snr_colors = {0.25: PAL["red"], 0.5: PAL["orange"], 1.0: PAL["blue"]}
snr_markers = {0.25: "^", 0.5: "D", 1.0: "o"}

for snr in [1.0, 0.5, 0.25]:
    cu, acc, se = get_acc_se("T2", snr)
    color = snr_colors[snr]
    marker = snr_markers[snr]

    # Smooth line
    try:
        p0 = [20, 100, 0.04, 50]
        bounds = ([0, 50, 0.001, 1], [80, 105, 1.0, 99])
        popt, _ = curve_fit(sigmoid, cu, acc, p0=p0, bounds=bounds, maxfev=10000)
        y_smooth = sigmoid(CU_smooth, *popt)
        se_interp = np.interp(CU_smooth, cu, se)
        ax.fill_between(CU_smooth, y_smooth - se_interp, y_smooth + se_interp,
                        color=color, alpha=0.08, zorder=1)
        ax.plot(CU_smooth, y_smooth, color=color, linewidth=2, zorder=2)
    except Exception:
        ax.plot(cu, acc, color=color, linewidth=2, zorder=2)

    ax.errorbar(cu, acc, yerr=se, fmt="none", ecolor=color,
                capsize=3, capthick=1, elinewidth=1, zorder=3, alpha=0.7)
    ax.scatter(cu, acc, color=color, marker=marker, s=55, zorder=4,
               edgecolors="white", linewidths=0.6,
               label=f"SNR = {SNR_NICE[snr]}")

ax.axhline(y=80, color=PAL["gray"], linestyle=":", linewidth=1, alpha=0.5)
ax.text(3, 81.5, "80% reliability", fontsize=8, color=PAL["gray"], alpha=0.7)

ax.set_xlabel("Context Utilization (CU, % of window)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("T2 (Two-Hop) Collapse — Effect of Noise Level")
ax.set_xlim(0, 100)
ax.set_ylim(-5, 110)
ax.legend(loc="upper right", frameon=True, fancybox=False)
ax.grid(True, axis="both", alpha=0.2)

save(fig, "fig5_t2_snr_comparison")


# ══════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to {OUT}/")
print("Files: fig1_hero_collapse, fig2_sigmoid_vs_linear, fig3_snr_effect,")
print("       fig4_calibration_scatter, fig5_t2_snr_comparison")
print("Formats: .pdf (vector) and .png (preview)")
