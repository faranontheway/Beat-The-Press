# Beat the Press: A Pressure-Aware Press Resistance Index (PRI) for Football

**Authors:** Farhan Zahin, Rafi Ali, Dept. of Computer Science and Engineering, BRAC University

The Press Resistance Index (PRI) measures how well a player keeps and progresses the ball **while under defensive pressure**, after adjusting for how difficult each situation was. The code uses only [StatsBomb Open Data](https://github.com/statsbomb/open-data).

## Data

8 men's competitions, 383 matches, ~703,000 Pass/Carry events, 153,678 pressured actions:

FIFA World Cup 2018 and 2022 · UEFA Euro 2020 and 2024 · Copa América 2024 · AFCON 2023 · La Liga 2020/21 · Bundesliga 2023/24

The notebook downloads the data automatically into `sb_data/` (about 1 GB). It is not stored in this repository.

## How to run

```bash
git clone https://github.com/<your-username>/beat-the-press.git
cd beat-the-press
pip install -r requirements.txt
jupyter notebook beat_the_press.ipynb      # then: Run All
```

- Runtime: about 10 minutes on a laptop (about 3 of those are the first download; later runs use the cache).
- Works on Google Colab too: upload the notebook and Run All.
- Tested with Python 3.12, pandas 3.0, scikit-learn 1.8, xgboost 3.4.
- Random seeds are fixed, so results match the tables below.

## Method in brief

1. **Retention label.** A pass is retained if completed. A carry is retained if the carrier's team still has the ball after it. If the carrier is **fouled**, the team keeps the ball, so the carry counts as retained.
2. **Expected retention.** A cross-validated XGBoost model (5 folds, grouped by match) predicts P(retained) from the situation: location, pass type/height/body part, recent opponent pressure, possession context. Player position is deliberately **not** a feature.
3. **Components** (per pressured action, averaged per player with shrinkage):
   - `OE` = retained − p̂ (observed minus expected retention)
   - `RVA` = OE × [xT(end) + opponent's xT at the mirrored spot] × 100 (extended OE: keeping the ball counts more where losing it is costly)
   - `threat` = xT gained, only if the ball was kept
   - `progression` = moved ≥ 10 units forward, only if the ball was kept
4. **Index.** OE⁺ = ½·z(OE) + ½·z(RVA), and

   **PRI = 0.4·z(OE⁺) + 0.3·z(threat) + 0.3·z(progression)**

   Outfield players with ≥ 50 pressured actions (871 players). Goalkeepers are ranked separately.

## Key results

| Expected-retention model | ROC-AUC | Brier |
|---|---|---|
| Previous version (balanced logistic regression) | 0.685 | 0.224 |
| **This version (calibrated XGBoost)** | **0.828** | **0.095** |

Leave-one-competition-out AUC: 0.813 – 0.841 across all 8 competitions.

| Average PRI by position | CB | FB | WM | AM | WING | DM | CM | FWD |
|---|---|---|---|---|---|---|---|---|
| Previous version | +0.03 | +0.11 | +0.25 | +0.14 | +0.18 | −0.36 | −0.29 | +0.08 |
| **This version** | **+0.44** | +0.19 | +0.15 | +0.02 | −0.04 | −0.07 | −0.17 | −0.63 |

Robustness:
- Re-weighting the three components: Spearman ρ ≥ 0.94 with the base ranking.
- Model without end-location features: AUC 0.780, PRI ranking ρ = 0.99 with the main model.
- Split-half reliability (random match halves): ρ ≈ 0.25. This is modest, because tournament data gives most players only 50–150 pressured actions.

## Repository contents

```
beat_the_press.ipynb      full pipeline: download → labels → xT → model → PRI → robustness
results/
  pri_v3_outfield.csv     PRI and components for all 871 eligible outfield players
  pri_by_position.csv/png average PRI by position group, previous vs this version
  top15_pri.png           top 15 players
  model_comparison.csv    previous vs new expected-retention model
  xT_grid.png             expected-threat grid
requirements.txt
LICENSE
```

## Limitations

- Pressure comes from StatsBomb's binary `under_pressure` tag. Defender positions (360 freeze-frames) are not used yet.
- xT is a 12 × 8 grid model, and retention is judged from the next event only.
- Match state, opponent quality and tactical instructions are not controlled for.
- Player samples are small (tournaments), which limits reliability.

PRI is a descriptive, difficulty-adjusted measure of pressure performance, not a fixed individual trait.

## Data attribution

Data provided by [StatsBomb](https://statsbomb.com/) through their free [open-data repository](https://github.com/statsbomb/open-data). The code in this repository is MIT-licensed. The data is subject to StatsBomb's own terms.
