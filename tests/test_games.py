import numpy as np
import pandas as pd
import pytest

from nflmodel.data import data_dir, raw_path
from nflmodel.games import GAME_COLUMNS, build_games, games_from_schedules, select_game_types


def _schedules(results, game_types=None):
    n = len(results)
    sched = pd.DataFrame({c: [None] * n for c in GAME_COLUMNS})
    sched["game_id"] = [f"g{i}" for i in range(n)]
    sched["season"] = 2020
    sched["gameday"] = [f"2020-09-{10 + i:02d}" for i in range(n)]
    sched["game_type"] = game_types or ["REG"] * n
    sched["home_score"] = [20 + r if r is not None else None for r in results]
    sched["away_score"] = [20 if r is not None else None for r in results]
    sched["result"] = results
    sched["extra_column"] = 1
    return sched


def test_home_win_handles_wins_losses_ties_and_unplayed():
    games = games_from_schedules(_schedules([7, -3, 0, None]))
    np.testing.assert_array_equal(games["home_win"], [1.0, 0.0, np.nan, np.nan])


def test_only_game_columns_kept():
    games = games_from_schedules(_schedules([7]))
    assert list(games.columns) == GAME_COLUMNS + ["home_win"]


def test_select_game_types_keeps_regular_season(cfg):
    games = games_from_schedules(_schedules([3, 3, 3], game_types=["REG", "WC", "SB"]))
    assert select_game_types(games, cfg)["game_type"].tolist() == ["REG"]


# ---- checks against the real snapshot (skipped on a fresh clone without data)
# Only seasons through 2024 are inspected; 2025 is the locked holdout.

@pytest.fixture
def real_games(cfg):
    if not (data_dir(cfg) / raw_path("schedules", None)).exists():
        pytest.skip("snapshot not built; run python -m scripts.build_data")
    games = build_games(cfg)
    return games[games["season"] <= 2024]


def test_one_row_per_game(real_games):
    assert real_games["game_id"].is_unique


def test_result_is_home_minus_away(real_games):
    diff = real_games["home_score"] - real_games["away_score"]
    assert (diff == real_games["result"]).all()


def test_spread_line_is_home_expected_margin(real_games):
    # Positive spread_line means the home team is favored, so it should track result upward
    # and home favorites should win most of their games.
    assert real_games["spread_line"].corr(real_games["result"]) > 0.3
    home_favs = real_games[real_games["spread_line"] > 3]
    assert home_favs["home_win"].mean() > 0.6
