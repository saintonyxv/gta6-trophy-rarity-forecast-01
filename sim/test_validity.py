"""Format and validity tests for the forecaster in ../main.py.

Hand-made cases cover the edge cases the brief calls out: empty history,
all-null trophies, single reads, rates at the clamps, scattered nulls,
horizon 0, duplicate timestamps, and a big 55-trophy case with a long
history (also the speed check). Every produced .out is validated against
the contest's shape rules. Run: python3 sim/test_validity.py
"""

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("forecaster", ROOT / "main.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
FAILURES = []


def check(cond, label):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"FAIL  {label}")
        FAILURES.append(label)


def validate(case, out, label):
    """The contest's shape rules, checked exactly."""
    raw = json.dumps(out, separators=(",", ":")).encode()
    check(len(raw) <= 65536, f"{label}: output <= 65536 bytes ({len(raw)})")
    check(set(out.keys()) == {"trophies"}, f"{label}: exactly one key 'trophies'")
    rows = out["trophies"]
    want_ids = [t["id"] for t in case["trophies"]]
    got_ids = [r["id"] for r in rows]
    check(sorted(got_ids) == sorted(want_ids) and len(set(got_ids)) == len(got_ids),
          f"{label}: one row per trophy id, each once")
    good = True
    for r in rows:
        if set(r.keys()) != {"id", "q"}:
            good = False
            break
        q = r["q"]
        if len(q) != 5:
            good = False
            break
        if any((not isinstance(x, (int, float))) or not math.isfinite(x) or x < 0 or x > 100 for x in q):
            good = False
            break
        if any(q[i] > q[i + 1] for i in range(4)):
            good = False
            break
    check(good, f"{label}: every q is 5 finite numbers in [0,100], non-decreasing")


def trophy(i, tier="bronze", points=15, secret=False):
    return {"id": i, "name": f"T{i}", "description": f"desc {i}", "tier": tier,
            "points": points, "secret": secret}


def read(at, rates, players=5000):
    return {"slot": at, "at": at, "players_tracked": players,
            "trophies_earned": 1000, "rates": rates}


def case(cid, trophies, history, horizon=1440, at="2026-11-20T00:00:00Z"):
    return {"schema": "makefun/gta6-trophies-case/v1", "id": cid, "at": at,
            "horizon_minutes": horizon, "target": "2026-11-21T00:00:00Z",
            "levels": LEVELS, "game": {"name": "Grand Theft Auto VI", "platform": "PS5"},
            "trophies": trophies, "history": history, "notes": "test"}


def run(c):
    return fc.forecast_case(c)


print("== edge cases ==")

# 1. normal small case, rates drifting up
c = case("t1", [trophy(1), trophy(2, "silver", 30)],
         [read("2026-11-19T00:00:00Z", [1.0, 5.0]),
          read("2026-11-19T12:00:00Z", [1.4, 5.6]),
          read("2026-11-20T00:00:00Z", [1.9, 6.1])])
out = run(c)
validate(c, out, "normal")
check(out["trophies"][0]["q"][2] > 1.9, "normal: median projects growth past the last read")

# 2. empty history
c = case("t2", [trophy(1), trophy(2)], [])
validate(c, run(c), "empty history")

# 3. trophy with all-null rates beside a normal one
c = case("t3", [trophy(1), trophy(2)],
         [read("2026-11-19T00:00:00Z", [None, 3.0]),
          read("2026-11-20T00:00:00Z", [None, 3.4])])
validate(c, run(c), "all-null trophy")

# 4. single read only
c = case("t4", [trophy(1)], [read("2026-11-20T00:00:00Z", [2.5])])
validate(c, run(c), "single read")

# 5. rates at the clamps: 0 and 100 in the reads
c = case("t5", [trophy(1), trophy(2)],
         [read("2026-11-19T00:00:00Z", [0.0, 100.0]),
          read("2026-11-20T00:00:00Z", [0.0, 100.0])])
out = run(c)
validate(c, out, "clamps")
check(out["trophies"][1]["q"][2] > 99.0, "clamps: a 100% trophy stays near 100")

# 6. scattered nulls mid-series
c = case("t6", [trophy(1)],
         [read("2026-11-18T00:00:00Z", [1.0]),
          read("2026-11-19T00:00:00Z", [None]),
          read("2026-11-19T12:00:00Z", [1.6]),
          read("2026-11-20T00:00:00Z", [None])])
validate(c, run(c), "scattered nulls")

# 7. horizon 0
c = case("t7", [trophy(1)],
         [read("2026-11-19T00:00:00Z", [2.0]), read("2026-11-20T00:00:00Z", [2.2])],
         horizon=0)
out = run(c)
validate(c, out, "horizon 0")
q = out["trophies"][0]["q"]
check(q[4] - q[0] < 0.5, "horizon 0: band is tight")

# 8. duplicate timestamps and a frozen page (identical reads)
c = case("t8", [trophy(1)],
         [read("2026-11-19T00:00:00Z", [1.1]) for _ in range(30)])
validate(c, run(c), "frozen page")

# 9. declining rates (dilution) project downward, not up
c = case("t9", [trophy(1)],
         [read("2026-11-18T00:00:00Z", [4.0]),
          read("2026-11-19T00:00:00Z", [3.2]),
          read("2026-11-20T00:00:00Z", [2.6])])
out = run(c)
validate(c, out, "declining")
check(out["trophies"][0]["q"][2] < 2.6, "declining: median follows the downward drift")

# 10. junk in the data: short rates list, a non-numeric rate, a read with no time
c = case("t10", [trophy(1), trophy(2)],
         [{"slot": "2026-11-19T00:00:00Z", "at": "2026-11-19T00:00:00Z",
           "players_tracked": 100, "trophies_earned": 5, "rates": [1.0]},
          {"slot": "2026-11-20T00:00:00Z", "at": "2026-11-20T00:00:00Z",
           "players_tracked": 100, "trophies_earned": 5, "rates": ["x", 7.0]}])
validate(c, run(c), "junk tolerance")

print("== determinism (through the real entry point) ==")
import os
import subprocess
import tempfile

kit_in = Path.home() / "workspace/makefun-kits/gta6-trophy-rarity-forecast-01-kit/input"
with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
    for d in (d1, d2):
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(kit_in), MAKEFUN_OUTPUT_DIR=d)
        subprocess.run([sys.executable, str(ROOT / "main.py")], env=env, check=True)
    for name in ("p00.out", "p01.out"):
        b1 = Path(d1, name).read_bytes()
        b2 = Path(d2, name).read_bytes()
        check(b1 == b2, f"determinism: {name} byte-identical across runs")

print("== speed: 55 trophies x 350 reads, final-case shape ==")
trophies = [trophy(i + 1) for i in range(55)]
history = []
base = 1763596800  # 2026-11-19T00:00:00Z-ish epoch; formatted below
from datetime import datetime, timezone, timedelta
t0 = datetime(2026, 11, 19, tzinfo=timezone.utc)
for j in range(350):
    at = (t0 + timedelta(minutes=10 * j)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rates = [round(0.4 + 4.5 * math.exp(-((i * 7) % 13) / 4.0) * (1 + 0.9 * j / 350.0), 2)
             for i in range(55)]
    history.append(read(at, rates, players=600 + 40 * j))
big = case("big", trophies, history, horizon=6510, at="2026-11-21T11:30:00Z")
start = time.perf_counter()
out = run(big)
elapsed = time.perf_counter() - start
validate(big, out, "big case")
check(elapsed < 8.0, f"speed: big case forecast in {elapsed:.2f}s (< 8s)")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    sys.exit(1)
print("all validity tests passed")
