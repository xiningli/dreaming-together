"""Generate the plots in PAPER.tex from results/grid.csv and results/grid_seeds.csv.

Outputs vector PDFs into figures/. Screenshot figures in figures/ are extracted
separately from demo/ videos (see PAPER.tex figure captions for provenance).

Run: python tools/make_paper_figures.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(0)
N_BOOT = 10_000

# Tol bright palette (colorblind-safe)
COLORS = {"A": "#4477AA", "B": "#EE6677", "C": "#228833", "NC": "#777777"}
LABELS = {
    "A": "A: FF + embedding",
    "B": "B: diffusion + embedding",
    "C": "C: diffusion + language",
    "NC": "NC: trained deaf",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def boot_ci(wins: np.ndarray) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(wins), size=(N_BOOT, len(wins)))
    means = wins[idx].mean(axis=1)
    return float(wins.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load(path: Path) -> dict[tuple, np.ndarray]:
    cells: dict[tuple, list[int]] = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["condition"], int(r["rate_ms"]), int(r["zeroed"]), int(r.get("seed", 1)) if "seed" in r and path.name == "grid_seeds.csv" else 1)
            cells[key].append(int(r["win"]))
    return {k: np.array(v) for k, v in cells.items()}


grid = load(ROOT / "results" / "grid.csv")          # seed 1 (+ NC)
seeds = load(ROOT / "results" / "grid_seeds.csv")   # seeds 2, 3

RATES = [50, 100, 250, 500, 1000, 2000]


def fig_rate_sweep() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for cond in "ABC":
        mean, lo, hi = [], [], []
        for rate in RATES:
            m, l, h = boot_ci(grid[(cond, rate, 0, 1)])
            mean.append(m); lo.append(l); hi.append(h)
        ax.plot(RATES, mean, "-o", ms=4, color=COLORS[cond], label=LABELS[cond])
        ax.fill_between(RATES, lo, hi, color=COLORS[cond], alpha=0.15, lw=0)
        # eval-time zeroed ablation at the trained rate
        mz, lz, hz = boot_ci(grid[(cond, 250, 1, 1)])
        ax.errorbar([250], [mz], yerr=[[mz - lz], [hz - mz]], fmt="x", ms=6,
                    color=COLORS[cond], alpha=0.8, capsize=2)
    m_nc, lo_nc, hi_nc = boot_ci(grid[("NC", 0, 1, 1)])
    ax.axhline(m_nc, color=COLORS["NC"], ls="--", lw=1)
    ax.axhspan(lo_nc, hi_nc, color=COLORS["NC"], alpha=0.12, lw=0)
    ax.text(700, m_nc - 0.033, "NC (trained deaf)", ha="center", fontsize=8, color=COLORS["NC"])
    ax.set_xscale("log")
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_xlabel(r"coordinator period $dt_{\mathrm{coord}}$ (ms), log scale")
    ax.set_ylabel("team win rate vs. fixed opponent")
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc="lower left", frameon=False)
    ax.annotate("$\\times$ = channel zeroed at eval (250 ms)", xy=(0.985, 0.03),
                xycoords="axes fraction", ha="right", fontsize=7.5, color="#444444")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_rate_sweep.pdf")
    plt.close(fig)


def seed_cell(cond: str, seed: int, zeroed: int) -> np.ndarray:
    if seed == 1:
        return grid[(cond, 250, zeroed, 1)]
    return seeds[(cond, 250, zeroed, seed)]


def fig_seeds() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    conds = ["A", "B", "C"]
    x = np.arange(3)
    seed_marks = {1: "o", 2: "s", 3: "^"}
    # panel 1: live win rate, one line per seed across conditions
    for seed in (1, 2, 3):
        ys, los, his = [], [], []
        for cond in conds:
            m, l, h = boot_ci(seed_cell(cond, seed, 0))
            ys.append(m); los.append(m - l); his.append(h - m)
        ax1.errorbar(x, ys, yerr=[los, his], fmt="-" + seed_marks[seed], ms=5,
                     color="#555555", mfc="white", capsize=2, lw=1,
                     label=f"seed {seed}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds)
    ax1.set_ylabel("win rate, live, 250 ms")
    ax1.set_title("(a) ordering per seed: C>B and A>B in 3/3")
    ax1.legend(frameon=False, loc="lower right")
    # panel 2: causal drop per condition per seed
    width = 0.25
    for j, seed in enumerate((1, 2, 3)):
        drops = []
        for cond in conds:
            live = seed_cell(cond, seed, 0).mean()
            zero = seed_cell(cond, seed, 1).mean()
            drops.append(100 * (live - zero))
        ax2.bar(x + (j - 1) * width, drops, width, color=[COLORS[c] for c in conds],
                alpha=(0.45, 0.7, 1.0)[j], edgecolor="white", lw=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(15, color="#888888", lw=0.8, ls=":")
    ax2.text(-0.42, 17.5, "G6 bar (+15)", fontsize=7, color="#666666")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conds)
    ax2.set_ylabel("causal drop (pts), live $-$ zeroed")
    ax2.set_title("(b) causal drop per seed (bars: seeds 1--3)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_seeds.pdf")
    plt.close(fig)


def fig_learnability() -> None:
    # P8 clone win rates from results/REPORT.md (single probe per cell, N=150
    # eval episodes; the raw logs are not in grid*.csv, hence literals here).
    p8 = {"C": {10: 0.907, 50: 0.847}, "B": {10: 0.667, 50: 0.653}}
    live = {"C": grid[("C", 250, 0, 1)].mean(), "B": grid[("B", 250, 0, 1)].mean()}
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for cond in ("C", "B"):
        ks = sorted(p8[cond])
        ax.plot(ks, [p8[cond][k] for k in ks], "-o", ms=5, color=COLORS[cond],
                label=LABELS[cond])
        ax.axhline(live[cond], color=COLORS[cond], ls="--", lw=1, alpha=0.6)
        ax.text(51, live[cond] + 0.006, f"{cond} original listener", fontsize=7,
                color=COLORS[cond], ha="right")
    ax.set_xticks([10, 50])
    ax.set_xlabel("K logged episodes given to fresh listener")
    ax.set_ylabel("clone-team win rate")
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False, loc="center right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_learnability.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_rate_sweep()
    fig_seeds()
    fig_learnability()
    print("wrote", *sorted(p.name for p in FIGDIR.glob("fig_*.pdf")))
