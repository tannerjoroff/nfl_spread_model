# NFL Spread Predictive Model

## Overview
This project builds an NFL margin-of-victory regression model (XGBoost) to identify potential spread inefficiencies. Predictions are converted into betting edges, calibrated to empirical cover probabilities using validation data, and evaluated using both flat betting diagnostics and conservative fractional Kelly sizing on a held-out test set.

## Data
- Source: `nflreadpy`
- Seasons: 2005–2025 (Regular Season)
- Unit of observation: Team-level game rows (home and away converted into team rows)

## Feature Engineering
- Base performance metrics (e.g., point differential, yards/play, turnovers, EPA/play)
- Rolling pre-game features (3-game window, `closed="left"` to avoid leakage)
- Week 1 handling via prior-season ending rolling stats
- Opponent-merged features and matchup deltas/ratios
- Nonlinear interaction features (e.g., power terms)

## Modeling
- Baseline: Linear Regression
- Final: XGBoost Regression predicting point differential

## Betting Framework
- Edge definition: `edge = predicted_home_margin - spread_line`
- Calibration: Validation-only edge bucketing + global shrinkage to stabilize probabilities
- Sizing: Fractional Kelly (e.g., 33% Kelly) with a max bet cap

## Evaluation
- Strict chronological train / validation / test split
- Expected vs realized units comparison
- Risk metrics: max drawdown in flat units and bankroll drawdown under Kelly sizing

## Key Results (Test Set Summary)
- Positive expected value out of sample (validation-calibrated)
- Realized performance positive but volatile
- Max flat-unit drawdown ≈ 13 units
- Max fractional Kelly bankroll drawdown ≈ 23%

## Notes
This project is for educational purposes only and does not constitute betting advice.