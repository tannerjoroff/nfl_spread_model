"""Leakage: changing what happened in game X must not change anything predicted for game X."""

import pandas as pd
import pytest

from nflmodel.models.elo import run_elo


@pytest.fixture
def season():
    # Six teams, three weeks, plus a second season.
    rows = [
        (2020, "2020-09-13", "A", "B", 7), (2020, "2020-09-13", "C", "D", -3), (2020, "2020-09-13", "E", "F", 10),
        (2020, "2020-09-20", "A", "C", 14), (2020, "2020-09-20", "B", "E", -6), (2020, "2020-09-20", "D", "F", 1),
        (2020, "2020-09-27", "A", "D", -10), (2020, "2020-09-27", "C", "E", 4), (2020, "2020-09-27", "F", "B", 21),
        (2021, "2021-09-12", "A", "B", 3), (2021, "2021-09-12", "C", "F", 0),
    ]
    return pd.DataFrame([
        {"game_id": f"g{i}", "season": s, "gameday": pd.Timestamp(d), "home_team": h,
         "away_team": a, "result": r, "location": "Home"}
        for i, (s, d, h, a, r) in enumerate(rows)
    ])


@pytest.mark.parametrize("x", ["g0", "g4", "g7", "g9"])
def test_changing_game_x_does_not_change_its_elo_or_earlier_games(season, x):
    before = run_elo(season).set_index("game_id")

    changed = season.copy()
    changed.loc[changed["game_id"] == x, "result"] = -changed.loc[changed["game_id"] == x, "result"] - 30
    after = run_elo(changed).set_index("game_id")

    day_x = season.loc[season["game_id"] == x, "gameday"].iloc[0]
    up_to_x = season.loc[season["gameday"] <= day_x, "game_id"]
    pd.testing.assert_frame_equal(before.loc[up_to_x], after.loc[up_to_x])


def test_changing_game_x_does_change_later_games(season):
    # Sanity check that the test above can fail: the outcome must matter afterwards.
    changed = season.copy()
    changed.loc[changed["game_id"] == "g0", "result"] = -40
    assert run_elo(season).loc[3, "elo_prob_home"] != run_elo(changed).loc[3, "elo_prob_home"]
