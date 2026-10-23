"""One row per game, from the home team's perspective.

nflverse sign conventions (checked against the data):
- result = home_score - away_score
- spread_line = the home team's expected margin, so positive means the home
  team is favored. This is the opposite of sportsbook notation, where a home
  favorite is listed as "-3".
"""

import numpy as np
import pandas as pd

from nflmodel.data import read_raw

GAME_COLUMNS = [
    "game_id", "season", "week", "game_type", "gameday", "location",
    "home_team", "away_team", "home_score", "away_score", "result",
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds",
]


def games_from_schedules(sched: pd.DataFrame) -> pd.DataFrame:
    """Select game-level columns and add home_win (1, 0, or NaN for a tie or unplayed game)."""
    games = sched[GAME_COLUMNS].copy()
    games["gameday"] = pd.to_datetime(games["gameday"])
    games["home_win"] = np.where(
        games["result"].isna() | (games["result"] == 0),
        np.nan,
        (games["result"] > 0).astype(float),
    )
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def build_games(cfg: dict) -> pd.DataFrame:
    """Every game in the snapshot seasons, playoffs included (filter with select_game_types)."""
    seasons = list(range(cfg["data"]["first_season"], cfg["data"]["last_season"] + 1))
    return games_from_schedules(read_raw("schedules", seasons=seasons, cfg=cfg))


def select_game_types(games: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep the game types used for training and evaluation (regular season only by default)."""
    return games[games["game_type"].isin(cfg["evaluation"]["game_types"])].reset_index(drop=True)
