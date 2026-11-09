"""Elo ratings with home field, a margin of victory multiplier, and regression between seasons.

Each team has one number. Before a game, the rating gap (plus home field) gives a
win probability. After it, both teams move toward what happened: the winner gains
what the loser loses, more when the result was a surprise and when the margin was
large. Settings follow the widely used FiveThirtyEight NFL model and are fixed, not
tuned, so the baseline cannot be accused of being fit to look good or bad.

Ratings for a game use only games played before it, so they cannot leak.
"""

import math
from dataclasses import dataclass

import pandas as pd

from nflmodel.evaluate import margin_metrics, prob_metrics

# Relocated franchises keep one rating across the move.
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0              # how far ratings move per game
    home_adv: float = 48.0       # Elo points added to the home team (none at neutral sites)
    revert: float = 1 / 3        # share of the gap to the mean removed between seasons
    mean: float = 1500.0         # starting rating and the level ratings revert toward
    elo_per_point: float = 25.0  # rating gap / 25 = predicted point margin


def win_prob(gap: float) -> float:
    """Probability that the side with a rating advantage of `gap` Elo points wins."""
    return 1 / (1 + 10 ** (-gap / 400))


def mov_multiplier(margin: float, winner_gap: float) -> float:
    """Scale the update by the margin of victory.

    log(margin + 1) grows with the margin but flattens out, so blowouts do not
    dominate. Dividing by (winner_gap * 0.001 + 2.2) shrinks the reward when a heavy
    favorite wins big, which keeps strong teams' ratings from inflating.
    A tie uses a margin of 1 and no gap correction.
    """
    if margin == 0:
        return math.log(2)
    return math.log(abs(margin) + 1) * 2.2 / (winner_gap * 0.001 + 2.2)


def run_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Walk through games in date order and return each game's pre-game Elo numbers.

    games needs game_id, season, gameday, location, home_team, away_team, result
    (home minus away; NaN for an unplayed game, which gets a prediction but no update).
    Returns game_id, elo_home_pre, elo_away_pre, elo_prob_home, elo_margin.
    """
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []

    for g in games.sort_values(["gameday", "game_id"]).itertuples(index=False):
        home, away = FRANCHISE.get(g.home_team, g.home_team), FRANCHISE.get(g.away_team, g.away_team)
        for team in (home, away):
            r = ratings.get(team, params.mean)
            if team in last_season and last_season[team] != g.season:
                r = params.mean + (1 - params.revert) * (r - params.mean)
            ratings[team], last_season[team] = r, g.season

        hfa = 0.0 if g.location == "Neutral" else params.home_adv
        gap = ratings[home] - ratings[away] + hfa
        p_home = win_prob(gap)
        rows.append((g.game_id, ratings[home], ratings[away], p_home, gap / params.elo_per_point))

        if pd.isna(g.result):
            continue
        actual = 1.0 if g.result > 0 else 0.0 if g.result < 0 else 0.5
        winner_gap = gap if g.result > 0 else -gap
        shift = params.k * mov_multiplier(g.result, winner_gap) * (actual - p_home)
        ratings[home] += shift
        ratings[away] -= shift

    return pd.DataFrame(rows, columns=["game_id", "elo_home_pre", "elo_away_pre", "elo_prob_home", "elo_margin"])


def score_elo(games: pd.DataFrame) -> dict:
    """Score Elo on games that already carry run_elo's columns (merged on game_id)."""
    margins = margin_metrics(games["result"], games["elo_margin"])
    probs = prob_metrics(games["home_win"], games["elo_prob_home"])
    return {
        "games": len(games),
        "ties": probs["n_ties"],
        "margin_mae": margins["mae"],
        "margin_rmse": margins["rmse"],
        "log_loss": probs["log_loss"],
        "brier": probs["brier"],
        "accuracy": probs["accuracy"],
    }
