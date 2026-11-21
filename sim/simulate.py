"""World simulator for the trophy rarity forecaster.

Generates synthetic Exophase-like worlds that follow the mechanics the kit
documents: the page only changes at Exophase refreshes (long frozen stretches,
then a jump), players tracked grows fast in launch week, and each trophy's
earners grow toward a per-trophy ceiling, so rates can rise (earning outruns
growth), saturate, or fall (dilution: tracked players grow faster than
earners). Reads sample the frozen page every 30 minutes early and every 10
minutes later, with rates printed to 2 decimals off integer earner counts.

Each world is cut into a case with the final's shape (about 2.5 days of
history, a 4.52-day horizon) and scored with the contest's exact loss --
mean pinball on the logit scale, rates clamped to [0.01, 99.99] -- for three
forecasters: this entry (../main.py), the house persistence baseline, and the
house curve baseline, both reimplemented here from the formulas the kit
documents.

Run: python3 sim/simulate.py [--worlds 60] [--seed 1]
"""

import argparse
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("forecaster", ROOT / "main.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
EPOCH = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)  # the final's checkpoint; relative day 0


def iso(days):
    return (EPOCH + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def pinball_loss(truth_rates, rows):
    """The contest's exact mean loss for one case."""
    total, count = 0.0, 0
    by_id = {r["id"]: r["q"] for r in rows}
    for tid, y in truth_rates.items():
        ly = logit(y)
        for tau, q in zip(LEVELS, by_id[tid]):
            d = ly - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


# --- house baselines, from the documented formulas ----------------------------

def persistence(case):
    hist = case["history"]
    last = hist[-1] if hist else {"rates": []}
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i] if i < len(last["rates"]) else None
        lg = logit(r if r is not None else 0.0)
        rows.append({"id": t["id"],
                     "q": [round(100.0 / (1.0 + math.exp(-(lg + z * spread))), 4) for z in Z]})
    return rows


def curve_baseline(case):
    """Logit vs ln(1 + hours since 2026-11-19T00:00Z), least squares over the
    last 24 hours of reads; spread = random walk and residual combined; fewer
    than 3 reads -> persistence."""
    hist = case["history"]
    if len(hist) < 3:
        return persistence(case)
    anchor = datetime(2026, 11, 19, tzinfo=timezone.utc)

    def hours(ts):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (dt - anchor).total_seconds() / 3600.0

    at_h = hours(case["at"])
    recent = [r for r in hist if hours(r["at"]) >= at_h - 24.0] or hist[-3:]
    if len(recent) < 3:
        return persistence(case)
    xs = [math.log1p(hours(r["at"])) for r in recent]
    x_t = math.log1p(at_h + case["horizon_minutes"] / 60.0)
    d = case["horizon_minutes"] / 1440.0
    rows = []
    for i, t in enumerate(case["trophies"]):
        pts = [(x, logit(r["rates"][i])) for x, r in zip(xs, recent) if r["rates"][i] is not None]
        if len(pts) < 3:
            r = hist[-1]["rates"][i]
            lg = logit(r if r is not None else 0.0)
            spread = 0.5 * math.sqrt(d)
            rows.append({"id": t["id"],
                         "q": [round(100.0 / (1.0 + math.exp(-(lg + z * spread))), 4) for z in Z]})
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        b = sxy / sxx if sxx > 0 else 0.0
        mu = my + b * (x_t - mx)
        resid = [p[1] - (my + b * (p[0] - mx)) for p in pts]
        rv = sum(r * r for r in resid) / max(1, n - 2)
        spread = math.sqrt(0.25 * d + rv)
        rows.append({"id": t["id"],
                     "q": [round(100.0 / (1.0 + math.exp(-(mu + z * spread))), 4) for z in Z]})
    return rows


# --- world generation -----------------------------------------------------------

REGIMES = ["standard", "hot", "slow", "dilution"]


def make_world(rng, regime):
    """Returns (case, truth) with the final's shape: ~2.48 days of history,
    4.52-day horizon, 55 trophies."""
    n_tr = 55
    mult_k = {"standard": 1.0, "hot": 1.6, "slow": 0.5, "dilution": 0.7}[regime]
    mult_n = {"standard": 1.0, "hot": 1.3, "slow": 0.6, "dilution": 1.8}[regime]

    # per-trophy ceilings (eventual earn fraction) and speeds
    ceilings, speeds = [], []
    for i in range(n_tr):
        if i == 0:
            ceilings.append(rng.uniform(0.002, 0.01))       # platinum
        elif i == 1:
            ceilings.append(1.0)                            # the automatic one
        elif i < 9:
            ceilings.append(rng.uniform(0.02, 0.18))        # gold / silver
        else:
            ceilings.append(rng.uniform(0.05, 0.75))        # bronze
        speeds.append(math.exp(rng.gauss(math.log(0.25), 0.55)) * mult_k)
    jump_sd = rng.uniform(0.04, 0.14)

    # players tracked: logistic growth from ~600 toward a launch-week ceiling
    n_max = rng.uniform(25000, 110000) * mult_n
    n_grow = rng.uniform(0.55, 0.95) * (1.15 if regime == "hot" else 1.0)
    players = 638.0
    earners = []
    for i in range(n_tr):
        start_frac = rng.uniform(0.02, 0.30)
        earners.append(ceilings[i] * players * start_frac if ceilings[i] < 1.0 else players)

    hist_start, checkpoint, target = -2.48, 0.0, 4.52
    refresh_gap_mean = rng.uniform(3.0, 8.0) / 24.0  # days

    # advance the latent state through refresh events, snapshotting the page
    snapshots = []  # (time_days, players, [rates percent exact])
    t = hist_start
    snapshots.append((t, players, [100.0 * e / players for e in earners]))
    while t < target + 0.1:
        gap = rng.expovariate(1.0 / refresh_gap_mean)
        t_next = t + gap
        # players grow logistically over the gap
        players = players + n_grow * players * (1.0 - players / n_max) * gap
        players = min(players, n_max)
        for i in range(n_tr):
            if ceilings[i] >= 1.0:
                earners[i] = players
                continue
            goal = ceilings[i] * players
            earners[i] = goal - (goal - earners[i]) * math.exp(-speeds[i] * gap)
            # unmodeled shock on the logit of the rate
            p = min(0.999999, max(1e-7, earners[i] / players))
            lg = math.log(p / (1 - p)) + rng.gauss(0.0, jump_sd)
            p2 = 1.0 / (1.0 + math.exp(-lg))
            earners[i] = p2 * players
        t = t_next
        snapshots.append((t, players, [100.0 * e / players for e in earners]))

    def page_at(when):
        state = snapshots[0]
        for s in snapshots:
            if s[0] <= when:
                state = s
            else:
                break
        return state

    # reads: 30-minute spacing for the first 5 hours, then every 10 minutes
    history = []
    rt = hist_start
    while rt <= checkpoint:
        _, n_at, rates_exact = page_at(rt)
        rates = []
        for i in range(n_tr):
            if rng.random() < 0.005:
                rates.append(None)
            else:
                rates.append(round(100.0 * round(rates_exact[i] / 100.0 * n_at) / n_at, 2))
        history.append({"slot": iso(rt), "at": iso(rt), "players_tracked": int(n_at),
                        "trophies_earned": int(sum(r for r in rates if r) * n_at / 100.0),
                        "rates": rates})
        rt += (0.5 / 24.0) if rt < hist_start + 5.0 / 24.0 else (10.0 / 1440.0)

    trophies = []
    tiers = ["platinum", "gold"] + ["silver"] * 6 + ["bronze"] * 46 + ["gold"]
    points = {"platinum": 300, "gold": 90, "silver": 30, "bronze": 15}
    for i in range(n_tr):
        tier = tiers[i] if i < len(tiers) else "bronze"
        trophies.append({"id": i + 1, "name": f"Trophy {i + 1}", "description": f"desc {i + 1}",
                         "tier": tier, "points": points[tier], "secret": False})

    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "f00", "at": iso(checkpoint),
            "horizon_minutes": 6510, "target": iso(target), "levels": LEVELS,
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19"},
            "trophies": trophies, "history": history, "notes": "simulated world"}

    _, _, truth_exact = page_at(target)
    truth = {i + 1: truth_exact[i] for i in range(n_tr)}
    return case, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    totals = {"mine": 0.0, "persistence": 0.0, "curve": 0.0}
    wins = {"vs_persistence": 0, "vs_curve": 0, "vs_best_house": 0}
    per_regime = {}
    for w in range(args.worlds):
        regime = REGIMES[w % len(REGIMES)]
        case, truth = make_world(rng, regime)
        losses = {
            "mine": pinball_loss(truth, fc.forecast_case(case)["trophies"]),
            "persistence": pinball_loss(truth, persistence(case)),
            "curve": pinball_loss(truth, curve_baseline(case)),
        }
        for k, v in losses.items():
            totals[k] += v
        if losses["mine"] < losses["persistence"]:
            wins["vs_persistence"] += 1
        if losses["mine"] < losses["curve"]:
            wins["vs_curve"] += 1
        if losses["mine"] < min(losses["persistence"], losses["curve"]):
            wins["vs_best_house"] += 1
        pr = per_regime.setdefault(regime, {"mine": 0.0, "persistence": 0.0, "curve": 0.0, "n": 0})
        for k, v in losses.items():
            pr[k] += v
        pr["n"] += 1

    n = args.worlds
    print(f"worlds: {n} (seed {args.seed})")
    print(f"mean logit pinball loss  mine        {totals['mine'] / n:.4f}")
    print(f"mean logit pinball loss  persistence {totals['persistence'] / n:.4f}")
    print(f"mean logit pinball loss  curve       {totals['curve'] / n:.4f}")
    print(f"worlds won: vs persistence {wins['vs_persistence']}/{n}, "
          f"vs curve {wins['vs_curve']}/{n}, vs best house {wins['vs_best_house']}/{n}")
    for regime, pr in sorted(per_regime.items()):
        m = pr["n"]
        print(f"  {regime:9s} n={m:2d}  mine {pr['mine'] / m:.4f}  "
              f"persistence {pr['persistence'] / m:.4f}  curve {pr['curve'] / m:.4f}")


if __name__ == "__main__":
    main()
