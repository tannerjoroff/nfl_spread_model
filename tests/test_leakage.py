"""Leakage: changing what happened in game X must not change anything predicted for game X."""

import numpy as np
import pandas as pd
import pytest

from nflmodel.features.team_stats import PBP_COLUMNS, team_game_stats, team_stat_features
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


# ---- play-by-play team ratings, end to end (raw plays -> features)


def _league(seed=0):
    """Four teams over two seasons; two games on each date, so same-day games exist."""
    rng = np.random.default_rng(seed)
    dates = ["2019-12-01", "2019-12-08", "2020-09-13", "2020-09-20", "2020-09-27"]
    pairs = [(("A", "B"), ("C", "D")), (("A", "C"), ("B", "D")), (("A", "D"), ("B", "C")),
             (("B", "A"), ("D", "C")), (("C", "A"), ("D", "B"))]
    games, plays = [], []
    for day, matchups in zip(dates, pairs):
        for home, away in matchups:
            gid = f"{day}_{away}_{home}"
            hs, as_ = (int(x) for x in rng.integers(0, 40, 2))
            games.append({"game_id": gid, "season": int(day[:4]), "gameday": pd.Timestamp(day),
                          "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
                          "result": hs - as_})
            for team in (home, away):
                for _ in range(20):
                    is_pass = rng.random() < 0.6
                    row = {c: 0 for c in PBP_COLUMNS}
                    row.update({"game_id": gid, "posteam": team, "pass": int(is_pass), "rush": int(not is_pass),
                                "epa": rng.normal(0, 1), "success": int(rng.random() < 0.45),
                                "yards_gained": int(rng.integers(-5, 30)), "yardline_100": int(rng.integers(1, 99)),
                                "fixed_drive": int(rng.integers(1, 12)), "fixed_drive_result": "Punt",
                                "penalty_team": None, "penalty_yards": np.nan})
                    plays.append(row)
    return pd.DataFrame(games), pd.DataFrame(plays)


def _features(games, pbp):
    tg = team_game_stats(pbp, games)
    return team_stat_features(games, tg, half_life=2, offseason_weight=0.5).set_index("game_id")


@pytest.mark.parametrize("x", [0, 3, 6, 9])
def test_changing_game_x_plays_and_score_leaves_its_features_alone(x):
    games, pbp = _league()
    before = _features(games, pbp)

    gid = games.loc[x, "game_id"]
    games2, pbp2 = games.copy(), pbp.copy()
    games2.loc[x, ["home_score", "away_score", "result"]] = [0, 60, -60]
    hit = pbp2["game_id"] == gid
    pbp2.loc[hit, "epa"] = pbp2.loc[hit, "epa"] + 5
    pbp2.loc[hit, "yards_gained"] = 80
    after = _features(games2, pbp2)

    # Game X and every game on or before its date (including same-day games) must not change.
    day = games.loc[x, "gameday"]
    up_to = games.loc[games["gameday"] <= day, "game_id"]
    pd.testing.assert_frame_equal(before.loc[up_to], after.loc[up_to])


def test_changing_an_early_game_does_change_later_features():
    games, pbp = _league()
    pbp2 = pbp.copy()
    pbp2.loc[pbp2["game_id"] == games.loc[0, "game_id"], "epa"] += 5
    later = games.loc[games["gameday"] > games.loc[0, "gameday"], "game_id"]
    assert not _features(games, pbp).loc[later].equals(_features(games, pbp2).loc[later])
