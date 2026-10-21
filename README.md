# NFL Game Model (rebuild in progress)

An NFL model that predicts margin of victory and win probability, then tests whether any betting strategy (spreads, moneylines, parlays) has an edge over the market. A weekly dashboard comes last.

**Status:** being rebuilt from scratch on the `rebuild` branch. There are no results yet.

## Why the rebuild

The original notebook (now in `legacy/`) reported a 9.1% test-set ROI. That result did not reproduce:

- The notebook no longer runs against current nflverse data (a schema change in team stats breaks a merge).
- The saved output came from cells run out of order.
- A clean top-to-bottom rerun with the original package versions gave 144 bets and a -7.4% bankroll. Across nine random seeds the result ranged from -12% to +1.4%.
- At the chosen threshold, flat betting on the validation set was -0.6 units over 190 bets, so validation never supported the betting rule.
- "9.1% ROI" was bankroll growth under Kelly sizing, not return per dollar wagered.
- The spread and total lines were model features, so the model largely learned the market line back.

The old notebook is kept for reference only. Nothing in the rebuild depends on it.

## Approach

- Reproducible pipeline: one command rebuilds everything from a hashed data snapshot, with seeded randomness.
- Walk-forward evaluation by season, with 2025 held out until the model and strategy list are frozen.
- Every result is compared against market baselines and reported with uncertainty. A result within about two standard errors of zero is reported as no evidence of edge.

Data sources, findings, and the experiment log are in `docs/RESEARCH.md`.

## Notes

This is a modeling project for educational purposes, not betting advice.
