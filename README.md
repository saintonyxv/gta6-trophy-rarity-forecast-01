[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saintonyxv/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saintonyxv/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Trophy Rarity Forecast

A forecaster for the earn rates of Grand Theft Auto VI's PlayStation 5 trophies on Exophase. For each case the program reads every recorded read of the game's trophy page before the checkpoint and writes, for every trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the target time.

The model works on the logit scale, where the contest's pinball loss lives. Because the page only changes at Exophase refreshes, each trophy's series is a step function: runs of equal rates are compressed to their endpoints, then fit twice — against time and against log-time — with recency-weighted least squares. The two projections are shrunk toward the case-wide median slope (a trophy with little history borrows the case's typical drift), damped, capped, and blended; the level anchors on a blend of the last read and the fits, and the spread adapts per trophy to its residual scatter and slope uncertainty. Everything is deterministic, standard-library-only Python, with no network access.

Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

## Run it

The grader runs the program once per case with that case's file in the input directory. Outside the grader's container, the directories come from the environment:

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces a `<id>.out` in the output directory.

Development checks live in `sim/`: `python3 sim/test_validity.py` runs format and edge-case tests, and `python3 sim/simulate.py --worlds 60` scores the forecaster against the house persistence and curve baselines on synthetic worlds built from the documented page mechanics.

## Referral

Find more bounties like this one on make.fun — referral link https://make.fun/r/hv020zhf (code hv020zhf).

## License

MIT — see LICENSE.
