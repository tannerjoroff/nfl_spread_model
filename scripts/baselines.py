"""Score the market and Elo baselines on every walk-forward test season.

    python -m scripts.baselines

Writes two files:
- results/baselines.csv: one row per model and test season, plus pooled rows.
- results/baseline_intervals.csv: bootstrap 95% intervals for the pooled comparisons,
  resampling whole weeks (primary) and single games, averaged over the config seeds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel.config import load_config, repo_root
from nflmodel.evaluate import bootstrap_ci, check_holdout, per_game_log_loss, walk_forward_splits
from nflmodel.games import build_games, select_game_types
from nflmodel.models.baselines import favorite_bets, market_margin, market_win_prob, score_market
from nflmodel.models.elo import run_elo, score_elo

RESULTS = Path("results") / "baselines.csv"
INTERVALS = Path("results") / "baseline_intervals.csv"


def paired_values(games: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-game values to bootstrap, each with its season-week cluster label.

    Elo-minus-market differences are paired: the same games, so game-to-game noise
    that affects both models cancels. Positive means Elo did worse than the market.
    """
    week = (games["season"] * 100 + games["week"]).to_numpy()
    p_mkt = market_win_prob(games)
    decided = (p_mkt.notna() & games["home_win"].notna()).to_numpy()
    y = games["home_win"].to_numpy()[decided]
    pm, pe = p_mkt.to_numpy()[decided], games["elo_prob_home"].to_numpy()[decided]

    margin_err_mkt = (games["result"] - market_margin(games)).abs()
    margin_err_elo = (games["result"] - games["elo_margin"]).abs()

    bets = favorite_bets(games)
    graded = bets[bets["won"].notna()]

    return {
        "elo_minus_market_log_loss": (per_game_log_loss(y, pe) - per_game_log_loss(y, pm), week[decided]),
        "elo_minus_market_brier": ((pe - y) ** 2 - (pm - y) ** 2, week[decided]),
        "elo_minus_market_margin_mae": ((margin_err_elo - margin_err_mkt).to_numpy(), week),
        "market_favorite_roi": (graded["profit"].to_numpy(), (graded["season"] * 100 + graded["week"]).to_numpy()),
    }


def interval_rows(games: pd.DataFrame, seeds: list[int]) -> list[dict]:
    rows = []
    for quantity, (values, weeks) in paired_values(games).items():
        for resample, clusters in (("week", weeks), ("game", None)):
            cis = [bootstrap_ci(values, clusters=clusters, seed=s) for s in seeds]
            los, his = [c["lo"] for c in cis], [c["hi"] for c in cis]
            rows.append({
                "quantity": quantity,
                "resample": resample,
                "estimate": cis[0]["estimate"],
                "se": np.mean([c["se"] for c in cis]),
                "lo": np.mean(los),
                "hi": np.mean(his),
                "seed_spread": max(max(los) - min(los), max(his) - min(his)),
                "n": cis[0]["n"],
                "clusters": cis[0]["n_clusters"],
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    test_seasons = [s.test_season for s in walk_forward_splits(cfg)]
    check_holdout(test_seasons, cfg)

    # Elo walks through every game (playoffs included) up to the last test season, nothing later.
    all_games = build_games(cfg)
    all_games = all_games[all_games["season"] <= max(test_seasons)]
    games = select_game_types(all_games, cfg).merge(run_elo(all_games), on="game_id", how="left")
    games = games[games["season"].isin(test_seasons)].reset_index(drop=True)

    pooled = f"{test_seasons[0]}-{test_seasons[-1]}"
    rows = []
    for label, g in [*((str(s), g) for s, g in games.groupby("season")), (pooled, games)]:
        rows.append({"model": "market", "season": label, **score_market(g)})
        rows.append({"model": "elo", "season": label, **score_elo(g)})
    results = pd.DataFrame(rows)
    intervals = pd.DataFrame(interval_rows(games, cfg["seeds"]))

    for df, path in ((results, RESULTS), (intervals, INTERVALS)):
        out = repo_root() / path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, float_format="%.4f", encoding="utf-8", lineterminator="\n")

    cols = ["model", "season", "margin_mae", "log_loss", "brier", "accuracy", "fav_roi"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results[cols].round(4).to_string(index=False))
        print()
        print(intervals.round(4).to_string(index=False))
    print(f"\nwrote {RESULTS.as_posix()} and {INTERVALS.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
