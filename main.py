"""Trophy rarity forecaster for gta6-trophy-rarity-forecast-01.

Reads every <id>.in case file in the input directory and writes <id>.out with,
for each of the case's trophies, five quantiles (levels 0.05, 0.25, 0.5, 0.75,
0.95) of its Exophase earn rate (percent) at the case's target time.

Model, in one paragraph: everything happens on the logit scale, where the
contest's pinball loss lives. The page a case is cut from only changes at
Exophase refreshes, so each trophy's read series is a step function; I compress
consecutive equal rates into single points, then fit each trophy's logit rate
twice -- against time and against log-time -- with recency-weighted least
squares. The two projections are damped, shrunk toward the case-wide median
slope (partial pooling, so a trophy with two reads borrows the case's typical
drift), capped, and blended. The level is anchored on a blend of the last read
and the fits' value at the last read, so one noisy read cannot dominate. The
spread grows with the square root of the horizon from the trophy's own last
read, widened by the trophy's residual scatter and by the slope's standard
error, so quiet trophies get tight bands and jumpy ones wide bands.

Deterministic: pure arithmetic on the case contents, no randomness, no clock,
no network. Python 3.13 standard library only.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader runs with -P, which leaves this file's directory off sys.path; put
# it back first so a sibling module would import (this entry is one file, but
# the line keeps local and graded runs identical).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OF_LEVEL = {tau: NormalDist().inv_cdf(tau) for tau in LEVELS}

RATE_LO = 0.01   # the scorer clamps rates into [0.01, 99.99] before the logit
RATE_HI = 99.99

# --- model constants (chosen on synthetic worlds, see sim/) -----------------
RECENCY_HALF_LIFE_DAYS = 0.5    # weight of a read halves this many days back
LOG_TIME_SCALE_HOURS = 6.0      # tau0 of the log-time transform ln(1 + h/tau0)
LINEAR_DAMP_DAYS = 3.0          # linear projection damped by 1/(1 + H/this)
BLEND_LINEAR = 0.35             # weight of the linear projection in the blend
DELTA_CAP = 1.15                # |projected logit change| cap over the horizon
DELTA_DAMP = 0.50               # global damping on the projected change
SLOPE_CAP_PER_DAY = 1.5         # |slope| cap, logit per day
LOG_SLOPE_CAP = 2.0             # |slope| cap, logit per unit of log-time
PRIOR_POOL = 3.0                # pooling: own slope weight k/(k + this)
SPREAD_BASE = 0.28              # diffusion per sqrt(day) before adaptation
SPREAD_RESID = 0.25             # extra spread per unit of residual sd
SPREAD_LO = 0.28
SPREAD_HI = 1.05
SPREAD_FLOOR = 0.06             # horizon-independent logit spread floor
LOW_RATE_SPREAD_BOOST = 1.15    # spread multiplier for rates under 0.5%
NO_DATA_SPREAD = 1.10           # per-sqrt-day spread when a trophy has no reads


# --- small numeric helpers ---------------------------------------------------

def clamp_rate(x):
    return min(RATE_HI, max(RATE_LO, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int so the JSON
    writes 100 rather than 100.0, matching the house style."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """ISO-8601 timestamp -> epoch seconds (float), or None if unparsable."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def weighted_linfit(ts, ys, ws):
    """Weighted least squares y ~ a + b*(t - t_ref) with t_ref the weighted mean
    of ts. Returns (a, b, resid_sd, se_b) or None when the fit is undefined
    (fewer than 2 points, no time span, or no weight)."""
    n = len(ts)
    if n < 2:
        return None
    wsum = sum(ws)
    if wsum <= 0:
        return None
    t_ref = sum(w * t for t, w in zip(ts, ws)) / wsum
    y_bar = sum(w * y for y, w in zip(ys, ws)) / wsum
    sxx = sum(w * (t - t_ref) ** 2 for t, w in zip(ts, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (t - t_ref) * (y - y_bar) for t, y, w in zip(ts, ys, ws))
    b = sxy / sxx
    a = y_bar  # value of the fit at t_ref
    resid = [y - (a + b * (t - t_ref)) for t, y in zip(ts, ys)]
    m = sum(1.0 for w in ws if w > 0)
    # dof-adjusted weighted residual variance (weights are relative, so the
    # slope's standard error rescales sxx by the mean weight)
    var = (sum(w * r * r for r, w in zip(resid, ws)) / wsum) * (m / max(1.0, m - 2.0))
    resid_sd = math.sqrt(max(0.0, var))
    se_b = math.sqrt(max(0.0, var) * wsum / (sxx * m))
    return a, b, resid_sd, se_b, t_ref


# --- per-trophy series --------------------------------------------------------

def trophy_points(history, idx):
    """(times_days, rates) for one trophy: non-null reads only, compressed as a
    step function -- each run of equal rates contributes its first and last
    time (one point if the run is a single read). A state that persisted for
    two days must say so: timestamping it only at its end would make the jump
    away from it look instantaneous and steep. Times are days relative to the
    case checkpoint, so they are <= 0."""
    pts = []
    run_start = None
    run_last = None
    run_rate = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        r = rates[idx]
        if r is None or not isinstance(r, (int, float)) or not math.isfinite(r):
            continue
        t = read.get("_t")
        if t is None:
            continue
        r = float(r)
        if run_rate is None or r != run_rate:
            if run_rate is not None:
                pts.append((run_start, run_rate))
                if run_last > run_start:
                    pts.append((run_last, run_rate))
            run_start = run_last = t
            run_rate = r
        else:
            run_last = t
    if run_rate is not None:
        pts.append((run_start, run_rate))
        if run_last > run_start:
            pts.append((run_last, run_rate))
    return pts


def fit_trophy(pts, case_start_days):
    """Fit one trophy's compressed points. Returns a dict with the fits against
    time and against log-time, or None when there are no points at all."""
    if not pts:
        return None
    ts = [p[0] for p in pts]
    ys = [logit(p[1]) for p in pts]
    t_last = ts[-1]
    ages = [t_last - t for t in ts]
    ws = [0.5 ** (age / RECENCY_HALF_LIFE_DAYS) for age in ages]
    lin = weighted_linfit(ts, ys, ws)
    # log-time: u = ln(1 + hours since the case's first read / scale)
    us = [math.log1p((t - case_start_days) * 24.0 / LOG_TIME_SCALE_HOURS) for t in ts]
    logfit = weighted_linfit(us, ys, ws)
    return {
        "n": len(pts),
        "t_last": t_last,
        "y_last": ys[-1],
        "last_rate": pts[-1][1],
        "lin": lin,
        "log": logfit,
        "u_last": us[-1],
    }


def shrunk_slope(b, se_b, k):
    """Shrink a fitted slope toward zero for estimation noise and thin data."""
    if b is None:
        return None
    noise = 1.0 / (1.0 + (se_b / 0.30) ** 2) if se_b is not None else 0.5
    thin = k / (k + 2.0)
    return b * noise * thin


# --- the forecast --------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or list(LEVELS)
    try:
        zs = [Z_OF_LEVEL.get(float(tau), NormalDist().inv_cdf(float(tau))) for tau in levels]
    except (TypeError, ValueError):
        levels, zs = list(LEVELS), [Z_OF_LEVEL[t] for t in LEVELS]

    at_sec = parse_time(case.get("at"))
    horizon_days = 0.0
    try:
        horizon_days = max(0.0, float(case.get("horizon_minutes") or 0.0) / 1440.0)
    except (TypeError, ValueError):
        horizon_days = 0.0

    # Stamp each read with its time in days relative to the checkpoint.
    stamped = []
    for read in history:
        if not isinstance(read, dict):
            continue
        sec = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if sec is None or at_sec is None:
            continue
        r2 = dict(read)
        r2["_t"] = (sec - at_sec) / 86400.0
        stamped.append(r2)
    stamped.sort(key=lambda r: r["_t"])
    case_start = stamped[0]["_t"] if stamped else 0.0

    fits = [fit_trophy(trophy_points(stamped, i), case_start) for i in range(len(trophies))]

    # Case-wide priors: median shrunk slope (per day) and median last rate.
    slopes = []
    last_rates = []
    for f in fits:
        if f is None:
            continue
        last_rates.append(f["last_rate"])
        if f["lin"] is not None:
            s = shrunk_slope(f["lin"][1], f["lin"][3], f["n"] - 1)
            if s is not None and math.isfinite(s):
                slopes.append(s)
    prior_slope = median(slopes) if slopes else 0.0
    prior_rate = median([clamp_rate(r) for r in last_rates]) if last_rates else 1.0

    u_target = math.log1p((horizon_days - case_start) * 24.0 / LOG_TIME_SCALE_HOURS)

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        tid = trophy.get("id") if isinstance(trophy, dict) else None
        rows.append({"id": tid, "q": forecast_one(f, horizon_days, u_target, prior_slope, prior_rate, zs)})
    return {"trophies": rows}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])


def forecast_one(f, horizon_days, u_target, prior_slope, prior_rate, zs):
    if f is None:
        # No reads at all for this trophy: the case-wide typical rate, wide band.
        mu = logit(prior_rate)
        sigma = NO_DATA_SPREAD * math.sqrt(max(horizon_days, 0.05)) + SPREAD_FLOOR
        return quantiles(mu, sigma, zs)

    h_i = max(0.0, horizon_days - f["t_last"])  # days from this trophy's last read
    k = f["n"] - 1  # distinct moves observed

    # --- projected change on the logit scale --------------------------------
    delta_lin = None
    if f["lin"] is not None:
        a, b, resid_sd, se_b, t_ref = f["lin"]
        own = shrunk_slope(b, se_b, k)
        lam = k / (k + PRIOR_POOL) if k > 0 else 0.0
        b_use = lam * own + (1.0 - lam) * prior_slope * 0.5
        b_use = max(-SLOPE_CAP_PER_DAY, min(SLOPE_CAP_PER_DAY, b_use))
        delta_lin = b_use * h_i / (1.0 + h_i / LINEAR_DAMP_DAYS)
        delta_lin = max(-1.4, min(1.4, delta_lin))
        fit_at_last_lin = a + b * (f["t_last"] - t_ref)
    else:
        fit_at_last_lin = None

    delta_log = None
    if f["log"] is not None:
        a2, b2, resid_sd2, se_b2, u_ref = f["log"]
        own2 = shrunk_slope(b2, se_b2, k)
        if own2 is not None:
            own2 = max(-LOG_SLOPE_CAP, min(LOG_SLOPE_CAP, own2))
            delta_log = own2 * max(0.0, u_target - f["u_last"])
            delta_log = max(-1.4, min(1.4, delta_log))
        fit_at_last_log = a2 + b2 * (f["u_last"] - u_ref)
    else:
        fit_at_last_log = None

    if delta_lin is not None and delta_log is not None:
        delta = BLEND_LINEAR * delta_lin + (1.0 - BLEND_LINEAR) * delta_log
    elif delta_lin is not None:
        delta = delta_lin
    elif delta_log is not None:
        delta = delta_log
    else:
        delta = prior_slope * 0.5 * h_i / (1.0 + h_i / LINEAR_DAMP_DAYS)
    delta = max(-DELTA_CAP, min(DELTA_CAP, delta * DELTA_DAMP))

    # --- level anchor ---------------------------------------------------------
    fit_levels = [v for v in (fit_at_last_lin, fit_at_last_log) if v is not None]
    if fit_levels:
        anchor = 0.45 * f["y_last"] + 0.55 * (sum(fit_levels) / len(fit_levels))
    else:
        anchor = f["y_last"]
    mu = anchor + delta

    # --- spread ---------------------------------------------------------------
    resid_sd = 0.0
    se_b = 0.0
    if f["lin"] is not None:
        resid_sd = f["lin"][2]
        se_b = f["lin"][3]
    per_day = min(SPREAD_HI, max(SPREAD_LO, SPREAD_BASE + SPREAD_RESID * resid_sd))
    var = per_day ** 2 * h_i + (0.6 * se_b * h_i) ** 2 + SPREAD_FLOOR ** 2
    sigma = math.sqrt(var)
    if f["last_rate"] < 0.5:
        sigma *= LOW_RATE_SPREAD_BOOST
    return quantiles(mu, sigma, zs)


def quantiles(mu, sigma, zs):
    out = []
    prev = 0.0
    for z in zs:
        q = 100.0 * sigmoid(mu + z * sigma)
        q = round4(min(100.0, max(0.0, q)))
        if q < prev:  # rounding must never break monotonicity
            q = prev
        out.append(q)
        prev = q
    return out


# --- entry point ---------------------------------------------------------------

def emergency_rows(case):
    """Last-resort valid output if anything above went wrong: every trophy at
    its last non-null rate (or a low prior), all five quantiles equal."""
    rows = []
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    for i, trophy in enumerate(trophies):
        rate = None
        for read in reversed(history):
            if isinstance(read, dict):
                rates = read.get("rates") or []
                if i < len(rates) and isinstance(rates[i], (int, float)) and math.isfinite(rates[i]):
                    rate = float(rates[i])
                    break
        if rate is None:
            rate = 1.0
        rate = min(100.0, max(0.0, rate))
        rows.append({"id": trophy.get("id") if isinstance(trophy, dict) else None,
                     "q": [round4(rate)] * 5})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not an object")
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = emergency_rows(case)
            except Exception:
                forecast = {"trophies": []}
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
