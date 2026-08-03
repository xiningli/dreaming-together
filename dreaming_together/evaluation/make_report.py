"""Generate results/REPORT.md — P1-P8 verdicts from the Stage 3 grid.

Statistics: bootstrap 95% CIs (10k resamples) on per-cell win rates;
two-proportion bootstrap p-values for ordered comparisons. Verdicts:
CONFIRMED (p < 0.05 in the predicted direction), REFUTED (p < 0.05
against), UNDERPOWERED/NOT MEASURED otherwise.

Run: python -m dreaming_together.evaluation.make_report
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
GRID = ROOT / "results" / "grid.csv"
SEEDS = ROOT / "results" / "grid_seeds.csv"
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


def load_seeds():
    """grid_seeds.csv: (condition, seed, zeroed) → rows, seeds 2-3 only.
    Seed 1 is the original bring-up stack and lives in grid.csv at
    (condition, 250, zeroed)."""
    cells = defaultdict(list)
    if not SEEDS.exists():
        return cells
    with open(SEEDS) as f:
        for r in csv.DictReader(f):
            key = (r["condition"], int(r["seed"]), int(r["zeroed"]))
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


def seed_run_dir(cond: str, seed: int) -> Path:
    if seed == 1:
        return ROOT / "runs" / ("stage2" if cond == "C" else f"stage2_{cond}")
    if cond == "A":
        return ROOT / "runs" / f"stage2_A_s{seed}"
    return ROOT / "runs" / f"stage2_{cond}_diff_s{seed}"


def g6_pass(cond: str, seed: int):
    """None if no gate file (e.g. condition A has no G6_RESULT_v2.json
    from the diffusion path); True/False otherwise."""
    run = seed_run_dir(cond, seed)
    for name in ("G6_RESULT_v2.json", "G6_RESULT.json"):
        p = run / name
        if p.exists():
            return bool(json.loads(p.read_text())["pass"])
    return None


def verdict(ok, p):
    if p < 0.05:
        return f"**CONFIRMED** (p={p:.4f})" if ok else f"**REFUTED** (p={p:.4f})"
    return f"UNDERPOWERED (p={p:.4f})"


def multi_seed_table(cells, seed_cells):
    """Per-seed win rate (live @250ms) and causal drop for A/B/C, seed
    1 pulled from grid.csv and seeds 2-3 from grid_seeds.csv. Returns
    (rows_by_cond_seed, markdown_lines). Seed is the unit of
    replication here — deliberately NOT pooled into one bootstrap,
    since pooling episodes across seeds would understate the
    between-seed variance the 3-seed requirement exists to expose."""
    per = {}
    for cond in ("A", "B", "C"):
        per[cond] = {}
        for seed in (1, 2, 3):
            if seed == 1:
                k_live, k_zero = (cond, 250, 0), (cond, 250, 1)
                src = cells
            else:
                k_live, k_zero = (cond, seed, 0), (cond, seed, 1)
                src = seed_cells
            if k_live not in src:
                continue
            w_live = wins(src[k_live])
            m_live, lo, hi = ci(w_live)
            drop = None
            if k_zero in src:
                drop = (m_live - wins(src[k_zero]).mean()) * 100
            per[cond][seed] = (m_live, lo, hi, drop)
    return per


def main() -> int:
    cells = load()
    seed_cells = load_seeds()
    L = []
    L.append("# Stage 3 Report — P1-P8 verdicts\n")
    L.append("**Evaluation protocol.** W is measured against a single "
             "frozen, calibrated scripted opponent (EliteScriptedTeam), "
             "identical for every cell. This replaces the design's "
             "original self-play W — a deliberate deviation, adopted "
             "because a fixed opponent removes co-evolution as a "
             "confound: every condition is graded against the same "
             "yardstick, so between-condition differences are "
             "attributable to the communication channel rather than to "
             "divergent opponent curricula. The cost is that W says "
             "nothing about inter-condition head-to-head play. Stacks "
             "are trained at dt_coord=250 ms and evaluated across the "
             "rate sweep without per-rate fine-tuning. 500 episodes/"
             "cell, bootstrap 95% CIs (10k resamples).\n")

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
    if ("NC", 0, 1) in cells:
        m, lo, hi = ci(wins(cells[("NC", 0, 1)]))
        L.append("| NC (trained deaf) | " + " | ".join(["—"] * len(rates))
                 + f" | {m:.3f} [{lo:.3f},{hi:.3f}] |")
    L.append("")
    if ("NC", 0, 1) in cells:
        w_nc = wins(cells[("NC", 0, 1)])
        L.append("**No-communication baseline.** NC is the same FF "
                 "listener architecture and training budget as condition "
                 "A, trained with the channel zeroed from the first "
                 "update. Contrast with the eval-time ablation "
                 "(zeroed@250 column): the ablation understates deaf "
                 "performance because those policies expect messages. "
                 "Gap analysis (condition live − NC = value of having a "
                 "channel; condition zeroed − NC = protocol-dependence "
                 "penalty):\n")
        for cond in ("A", "B", "C"):
            if (cond, 250) in W and (cond, 250, 1) in cells:
                gap_live = W[(cond, 250)].mean() - w_nc.mean()
                gap_abl = wins(cells[(cond, 250, 1)]).mean() - w_nc.mean()
                p_live = p_greater(W[(cond, 250)], w_nc)
                L.append(f"- {cond}: channel value {gap_live:+.3f} "
                         f"({verdict(gap_live > 0, p_live)}); "
                         f"ablation-vs-NC {gap_abl:+.3f}")
        L.append("")

    # P1 — cross-seed consistency is the authoritative verdict once all
    # 3 seeds are evaluated; the single-seed bootstrap (below) is
    # supporting detail, not the headline, once replication exists.
    per = multi_seed_table(cells, seed_cells)
    seeds_present = sorted({s for cond in per for s in per[cond]})
    orderings = []
    for seed in (1, 2, 3):
        if all(seed in per[c] for c in "ABC"):
            wc, wb, wa = (per[c][seed][0] for c in "CBA")
            orderings.append((seed, wc > wb, wb > wa, wc > wa))

    L.append("## Verdicts\n")
    if all((c, 250) in W for c in "ABC"):
        p_cb = p_greater(W[("C", 250)], W[("B", 250)])
        p_ba = p_greater(W[("B", 250)], W[("A", 250)])
        wc, wb, wa = (W[(c, 250)].mean() for c in "CBA")
        if len(orderings) >= 3:
            n_cb = sum(1 for _, cb, _, _ in orderings if cb)
            n_ba = sum(1 for _, _, ba, _ in orderings if ba)
            n = len(orderings)
            if n_cb == n and n_ba == n:
                overall = "**CONFIRMED** (replicated across all seeds)"
            elif n_cb == n and n_ba == 0:
                overall = (f"**PARTIALLY CONFIRMED, REPLICATED** — C>B "
                           f"holds in {n_cb}/{n} independent seeds; B>A "
                           f"is **REFUTED**, reversing (A>B) in {n}/{n} "
                           f"seeds")
            elif n_cb == n:
                overall = (f"**PARTIALLY CONFIRMED, REPLICATED** — C>B "
                           f"holds in {n_cb}/{n} seeds; B>A holds in "
                           f"{n_ba}/{n}")
            else:
                overall = (f"**NOT CONSISTENTLY CONFIRMED** — C>B holds "
                           f"in only {n_cb}/{n} seeds")
            L.append(f"**P1** W(C,250) > W(B,250) > W(A,250), evaluated "
                     f"across {n} independent training seeds each: "
                     f"{overall}. (Seed-1 point estimate: C={wc:.3f}, "
                     f"B={wb:.3f}, A={wa:.3f}; single-seed bootstrap "
                     f"C>B {verdict(wc > wb, p_cb)}, B>A "
                     f"{verdict(wb > wa, p_ba)} — see the per-seed table "
                     f"below for why a pooled bootstrap is not the "
                     f"right statistic here.)\n")
        else:
            both = p_cb < 0.05 and p_ba < 0.05
            L.append(f"**P1** W(C,250) > W(B,250) > W(A,250): observed "
                     f"C={wc:.3f}, B={wb:.3f}, A={wa:.3f}. C>B: "
                     f"{verdict(wc > wb, p_cb)}; B>A: {verdict(wb > wa, p_ba)}. "
                     f"P1 overall (single seed — replication pending): "
                     f"{'**CONFIRMED**' if both and wc > wb > wa else ('**PARTIALLY CONFIRMED** (C>B holds; B>A ' + ('reversed' if wa > wb else 'not significant') + ')' if p_cb < 0.05 and wc > wb else 'NOT CONFIRMED')}.\n")

    # P1 multi-seed replication (supporting detail)
    L.append("### P1 multi-seed replication\n")
    if len(seeds_present) < 2:
        L.append("Only seed 1 evaluated so far — seeds 2-3 pending "
                 "(`evaluation/stage3_seeds.py`); the ordering above is "
                 "a single-seed result. See caveats.\n")
    else:
        L.append("Win rate (live, 250 ms) and causal drop (pts) per "
                 "seed. Seed is treated as the unit of replication: "
                 "each seed is one independent training run, so its "
                 "500-episode win rate is one data point, not pooled "
                 "with the others — with only 2-3 seeds the honest "
                 "summary is the ordering's consistency across runs, "
                 "not a bootstrap p-value over pooled episodes.\n")
        L.append("| seed | " + " | ".join(f"W(cond={c})/drop" for c in "ABC")
                 + " |")
        L.append("|---|---|---|---|")
        for seed in (1, 2, 3):
            row = [str(seed)]
            for cond in "ABC":
                if seed in per[cond]:
                    m, lo, hi, drop = per[cond][seed]
                    dtxt = f"{drop:+.1f}" if drop is not None else "—"
                    row.append(f"{m:.3f} [{lo:.3f},{hi:.3f}] / {dtxt}")
                else:
                    row.append("—")
            L.append("| " + " | ".join(row) + " |")
        if orderings:
            n_cb = sum(1 for _, cb, _, _ in orderings if cb)
            n_ba = sum(1 for _, _, ba, _ in orderings if ba)
            L.append(f"\nAcross {len(orderings)} fully-evaluated seeds: "
                     f"C>B held in {n_cb}/{len(orderings)}; B>A held in "
                     f"{n_ba}/{len(orderings)}. Per-seed detail: "
                     + "; ".join(f"seed{s}: C>B={cb}, B>A={ba}"
                                 for s, cb, ba, _ in orderings) + ".\n")
        gate_lines = []
        for cond in "ABC":
            for seed in sorted(per.get(cond, {})):
                ok = g6_pass(cond, seed)
                if ok is not None:
                    gate_lines.append(f"{cond}s{seed}={'PASS' if ok else 'FAIL'}")
        if gate_lines:
            n_fail = sum(1 for g in gate_lines if "FAIL" in g)
            L.append(f"\nPer-seed G6 gate (pre-registered win/diversity/"
                     f"causal-drop bar), disclosed regardless of "
                     f"outcome — {n_fail}/{len(gate_lines)} seed-stacks "
                     f"failed it even though all seeds are included in "
                     f"the ordering above: " + ", ".join(gate_lines)
                     + ". A seed failing G6 (e.g. a negative causal "
                     "drop) is a real training-run outcome, not grounds "
                     "for exclusion or a rerun-until-it-passes policy.\n")
    L.append("")

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
    L.append("**P5** Γ self/world discrimination: **DROPPED** — the Γ "
             "instrument (contact vs non-contact denoising-error "
             "analysis) was not built; the prediction is withdrawn from "
             "this study's claims rather than deferred.\n")
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
    n_seeds = len(seeds_present) if seeds_present else 1
    seed_caveat = (
        f"- {n_seeds} training seed(s) per condition evaluated at "
        f"250 ms (P1); the design's 3-seed requirement is "
        f"{'met' if n_seeds >= 3 else 'partially met' if n_seeds > 1 else 'NOT met'} "
        "for the headline ordering. The rate sweep (P2, P3, P6, P7) and "
        "BPW (P4) remain seed-1 only — replicating those across seeds "
        "was out of scope for this pass.\n")
    L.append("- Fixed-opponent W, not self-play W: every condition faces "
             "the same frozen scripted opponent. Removes co-evolution "
             "confounds; does not measure inter-condition head-to-head.\n"
             + seed_caveat +
             "- Stacks trained at 250 ms only; rate sweep is evaluation-"
             "time (per-rate fine-tunes were cut per the scope ladder).\n"
             "- P5 dropped, not deferred — no instrument was built and "
             "none is claimed. P7 proxy only.\n"
             "- Diffusion conditions use the sample-and-select "
             "architecture (frozen prior + learned scorer) — G4's "
             "pre-registered fallback.\n")
    OUT.write_text("\n".join(L))
    print(f"→ {OUT}")
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
