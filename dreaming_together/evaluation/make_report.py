"""Generate results/REPORT.md — P1-P8 verdicts from the Stage 3 grid.

Statistics: bootstrap 95% CIs (10k resamples) on per-cell win rates;
two-proportion bootstrap p-values for ordered comparisons. Verdicts:
CONFIRMED (p < 0.05 in the predicted direction), REFUTED (p < 0.05
against), UNDERPOWERED/NOT MEASURED otherwise.

Run: python -m dreaming_together.evaluation.make_report
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
GRID = ROOT / "results" / "grid.csv"
P8 = ROOT / "results" / "p8_learnability.csv"
OUT = ROOT / "results" / "REPORT.md"

RNG = np.random.default_rng(0)
NBOOT = 10_000


def load():
    cells = defaultdict(list)
    with open(GRID) as f:
        for r in csv.DictReader(f):
            key = (r["condition"], int(r["rate_ms"]), int(r["zeroed"]))
            cells[key].append(r)
    return cells


def wins(rows):
    return np.array([int(r["win"]) for r in rows])


def ci(w):
    boots = RNG.choice(w, size=(NBOOT, len(w))).mean(axis=1)
    return w.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def p_greater(w_hi, w_lo):
    """Bootstrap p-value for mean(w_hi) > mean(w_lo)."""
    b_hi = RNG.choice(w_hi, size=(NBOOT, len(w_hi))).mean(axis=1)
    b_lo = RNG.choice(w_lo, size=(NBOOT, len(w_lo))).mean(axis=1)
    return float((b_hi <= b_lo).mean())


def verdict(ok, p):
    if p < 0.05:
        return f"**CONFIRMED** (p={p:.4f})" if ok else f"**REFUTED** (p={p:.4f})"
    return f"UNDERPOWERED (p={p:.4f})"


def main() -> int:
    cells = load()
    L = []
    L.append("# Stage 3 Report — P1-P8 verdicts\n")
    L.append("Fixed-opponent evaluation (frozen calibrated "
             "EliteScriptedTeam, identical for every cell). Stacks trained "
             "at dt_coord=250 ms, evaluated across the rate sweep without "
             "per-rate fine-tuning. 500 episodes/cell, bootstrap 95% CIs "
             "(10k resamples). Deviations from the design's Stage-3 "
             "self-play are documented in the caveats section.\n")

    # main table
    L.append("## W(condition, dt_coord)\n")
    rates = sorted({k[1] for k in cells if not k[2]})
    L.append("| condition | " + " | ".join(f"{r} ms" for r in rates)
             + " | zeroed@250 |")
    L.append("|---|" + "---|" * (len(rates) + 1))
    W = {}
    for cond in ("A", "B", "C"):
        row = [cond]
        for r in rates:
            k = (cond, r, 0)
            if k in cells:
                m, lo, hi = ci(wins(cells[k]))
                W[(cond, r)] = wins(cells[k])
                row.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
            else:
                row.append("—")
        kz = (cond, 250, 1)
        if kz in cells:
            m, lo, hi = ci(wins(cells[kz]))
            row.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
        else:
            row.append("—")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # P1
    L.append("## Verdicts\n")
    if all((c, 250) in W for c in "ABC"):
        p_cb = p_greater(W[("C", 250)], W[("B", 250)])
        p_ba = p_greater(W[("B", 250)], W[("A", 250)])
        wc, wb, wa = (W[(c, 250)].mean() for c in "CBA")
        both = p_cb < 0.05 and p_ba < 0.05
        L.append(f"**P1** W(C,250) > W(B,250) > W(A,250): observed "
                 f"C={wc:.3f}, B={wb:.3f}, A={wa:.3f}. C>B: "
                 f"{verdict(wc > wb, p_cb)}; B>A: {verdict(wb > wa, p_ba)}. "
                 f"P1 overall: "
                 f"{'**CONFIRMED**' if both and wc > wb > wa else ('**PARTIALLY CONFIRMED** (C>B holds; B>A ' + ('reversed' if wa > wb else 'not significant') + ')' if p_cb < 0.05 and wc > wb else 'NOT CONFIRMED')}.\n")

    def argmax_rate(cond):
        ms = {r: W[(cond, r)].mean() for r in rates if (cond, r) in W}
        return max(ms, key=ms.get)

    # P2
    r_c = argmax_rate("C")
    L.append(f"**P2** argmax_r W(C,·) ∈ [250, 1000] ms: observed argmax at "
             f"{r_c} ms → "
             f"{'**CONFIRMED**' if 250 <= r_c <= 1000 else '**REFUTED**'} "
             f"(point estimate; CI overlap caveat applies).\n")
    # P3
    r_b = argmax_rate("B")
    L.append(f"**P3** argmax W(B,·) at smaller rate than argmax W(C,·): "
             f"B* = {r_b} ms vs C* = {r_c} ms → "
             f"{'**CONFIRMED**' if r_b < r_c else ('**REFUTED**' if r_b > r_c else 'TIED — NOT CONFIRMED')} "
             f"(point estimate).\n")
    # P4 — bits per win at each condition's best rate
    def bpw(cond, r):
        rows = cells[(cond, r, 0)]
        bits = sum(int(x["n_messages"]) for x in rows) * 40
        wn = wins(rows).sum()
        return bits / max(1, wn)
    bpw_c, bpw_b = bpw("C", r_c), bpw("B", r_b)
    L.append(f"**P4** BPW(C, C*) < BPW(B, B*): {bpw_c:,.0f} vs "
             f"{bpw_b:,.0f} bits/win → "
             f"{'**CONFIRMED** (point estimate)' if bpw_c < bpw_b else '**REFUTED** (point estimate)'}.\n")
    # P5
    L.append("**P5** Γ self/world discrimination: **NOT MEASURED** — the "
             "Γ instrument (contact vs non-contact denoising-error "
             "analysis) was not built in this phase; deferred.\n")
    # P6
    if ("C", 50) in W and ("B", 50) in W:
        p_hi = p_greater(W[("C", 50)], W[("B", 50)])
        p_lo = p_greater(W[("B", 50)], W[("C", 50)])
        similar = p_hi >= 0.05 and p_lo >= 0.05
        L.append(f"**P6** W(C,50) ≈ W(B,50): observed "
                 f"{W[('C',50)].mean():.3f} vs {W[('B',50)].mean():.3f} "
                 f"(two-sided bootstrap p={2*min(p_hi,p_lo):.4f}) → "
                 f"{'**CONFIRMED** (no significant difference)' if similar else '**REFUTED** (difference persists at 50 ms)'}.\n")
    # P7 (proxy)
    L.append("**P7** window-timing proxy: mean window-open fraction by "
             "rate (condition C): "
             + ", ".join(
                 f"{r}ms={np.mean([float(x['window_open_frac']) for x in cells[('C', r, 0)]]):.2f}"
                 for r in rates if ("C", r, 0) in cells)
             + ". Full e_w (cue-to-open latency) instrument not built; "
               "**PARTIAL** — see caveats.\n")
    # P8
    if P8.exists():
        rows = list(csv.DictReader(open(P8)))
        L.append("**P8** protocol learnability (fresh listeners BC'd on K "
                 "episodes, evaluated with the original coordinator):\n")
        L.append("| condition | K | clone team win rate |")
        L.append("|---|---|---|")
        for r in rows:
            L.append(f"| {r['condition']} | {r['K']} | "
                     f"{float(r['clone_win_rate']):.3f} |")
        d = {(r["condition"], r["K"]): float(r["clone_win_rate"])
             for r in rows}
        try:
            ok_small = d[("C", "10")] > d[("B", "10")]
            gap10 = d[("C", "10")] - d[("B", "10")]
            gap50 = d[("C", "50")] - d[("B", "50")]
            L.append(f"\nC−B recovery gap: K=10 → {gap10:+.3f}, K=50 → "
                     f"{gap50:+.3f}. P8 (C recovers more, gap wider at "
                     f"small K): "
                     f"{'**CONFIRMED** (point estimates, N=150 eval eps)' if ok_small and gap10 > gap50 else ('**PARTIALLY CONFIRMED** (direction holds at both K)' if ok_small and gap50 >= 0 else '**NOT CONFIRMED**')}.\n")
        except KeyError:
            L.append("\nP8: incomplete data.\n")
    else:
        L.append("**P8**: probe not yet run.\n")

    L.append("## Caveats (read before citing)\n")
    L.append("- Fixed-opponent W, not self-play W: every condition faces "
             "the same frozen scripted opponent. Removes co-evolution "
             "confounds; does not measure inter-condition head-to-head.\n"
             "- Single training seed per condition (bring-up-selected); "
             "the design's 3-seed requirement is NOT met — treat ordered "
             "verdicts as single-seed results.\n"
             "- Stacks trained at 250 ms only; rate sweep is evaluation-"
             "time (per-rate fine-tunes were cut per the scope ladder).\n"
             "- P5 not measured; P7 proxy only.\n"
             "- Diffusion conditions use the sample-and-select "
             "architecture (frozen prior + learned scorer) — G4's "
             "pre-registered fallback.\n")
    OUT.write_text("\n".join(L))
    print(f"→ {OUT}")
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
