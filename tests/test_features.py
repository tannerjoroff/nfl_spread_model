import numpy as np
import pandas as pd
import pytest

from nflmodel.features.team_stats import (
    GROUPS,
    PBP_COLUMNS,
    RATES,
    _cumulative_rates,
    decayed_ratings,
    team_game_stats,
    team_stat_features,
)


def play(team, **kw):
    row = {c: 0 for c in PBP_COLUMNS}
    row.update({"game_id": "g1", "posteam": team, "epa": np.nan, "success": 0, "yards_gained": 0,
                "yardline_100": 50, "fixed_drive": 1, "fixed_drive_result": "Punt",
                "penalty_team": None, "penalty_yards": np.nan})
    row.update(kw)
    return row


def one_game(home="A", away="B"):
    return pd.DataFrame({"game_id": ["g1"], "season": [2020], "gameday": [pd.Timestamp("2020-09-13")],
                         "home_team": [home], "away_team": [away], "home_score": [24], "away_score": [17],
                         "result": [7]})


@pytest.fixture
def stats():
    pbp = pd.DataFrame([
        # A: a 25-yard pass, a run that converts a third down, a sack, a failed third down
        play("A", **{"pass": 1, "epa": 0.5, "success": 1, "yards_gained": 25}),
        play("A", rush=1, epa=-0.2, yards_gained=3, third_down_converted=1),
        play("A", **{"pass": 1, "epa": -1.0, "yards_gained": -7, "sack": 1, "third_down_failed": 1}),
        play("A", qb_kneel=1, epa=-0.1),                         # kneel: not a play
        play("A", **{"pass": 1, "qb_spike": 1, "epa": -0.3}),    # spike: not a play
        play("A", penalty=1, penalty_team="A", penalty_yards=10),
        # A red zone: one drive ends in a touchdown, another in a field goal (two plays each)
        play("A", yardline_100=15, fixed_drive=2, fixed_drive_result="Touchdown"),
        play("A", yardline_100=8, fixed_drive=2, fixed_drive_result="Touchdown"),
        play("A", yardline_100=12, fixed_drive=3, fixed_drive_result="Field goal"),
        play("A", yardline_100=12, fixed_drive=3, fixed_drive_result="Field goal"),
        # B: a 12-yard run and an interception
        play("B", rush=1, epa=0.3, success=1, yards_gained=12),
        play("B", **{"pass": 1, "epa": -2.0, "interception": 1}),
    ])
    return team_game_stats(pbp, one_game()).set_index("team")


def test_offense_totals_by_hand(stats):
    a = stats.loc["A"]
    assert a["off_plays"] == 3 and a["off_pass_plays"] == 2 and a["off_rush_plays"] == 1
    assert a["off_epa"] == pytest.approx(-0.7)
    assert a["off_pass_epa"] == pytest.approx(-0.5)
    assert a["off_explosive"] == 1 and a["off_sacks"] == 1
    assert a["off_third_conv"] == 1 and a["off_third_att"] == 2
    assert a["off_rz_trips"] == 2 and a["off_rz_td"] == 1
    assert a["off_penalty_yards"] == 10
    assert a["off_points"] == 24 and a["off_games"] == 1


def test_defense_is_the_opponents_offense(stats):
    a, b = stats.loc["A"], stats.loc["B"]
    assert a["def_plays"] == 2 and a["def_turnovers"] == 1 and a["def_explosive"] == 1
    assert a["def_epa"] == pytest.approx(-1.7)
    assert a["def_points"] == 17
    assert b["def_epa"] == pytest.approx(a["off_epa"])


def test_relocated_teams_use_current_code():
    pbp = pd.DataFrame([play("LA", rush=1, epa=0.1), play("B", rush=1, epa=0.2)])
    stats = team_game_stats(pbp, one_game(home="STL"))
    assert set(stats["team"]) == {"LA", "B"}
    assert stats.set_index("team").loc["LA", "off_epa"] == pytest.approx(0.1)


# ---- decay and league-relative ratings

def team_games(rows):
    """rows: (game_id, season, gameday, team, off_epa). Every count is 1, every other total 0."""
    out = []
    for gid, season, day, team, epa in rows:
        r = {"game_id": gid, "season": season, "gameday": pd.Timestamp(day), "team": team}
        for side in ("off", "def"):
            for stat, denom in RATES.items():
                r[f"{side}_{stat}"], r[f"{side}_{denom}"] = 0.0, 1.0
        r["off_epa"] = epa
        out.append(r)
    return pd.DataFrame(out)


def test_newer_games_count_more():
    tg = team_games([("g1", 2020, "2020-09-13", "A", 1.0), ("g2", 2020, "2020-09-20", "A", 3.0)])
    cum = _cumulative_rates(tg, half_life=1, offseason_weight=1.0)
    # with a half-life of one game, the older game counts half: (0.5 * 1 + 3) / 1.5
    assert cum["off_epa"].iloc[-1] == pytest.approx((0.5 * 1 + 3) / 1.5)


def test_offseason_weight_discounts_earlier_seasons():
    tg = team_games([("g1", 2019, "2019-12-01", "A", 2.0), ("g2", 2020, "2020-09-13", "A", 0.0)])
    cum = _cumulative_rates(tg, half_life=1e9, offseason_weight=0.5)
    assert cum["off_epa"].iloc[-1] == pytest.approx((0.5 * 2) / 1.5)


def test_ratings_use_only_earlier_games_and_are_league_relative():
    tg = team_games([
        ("g1", 2020, "2020-09-13", "A", 1.0), ("g1", 2020, "2020-09-13", "B", 0.0),
        ("g2", 2020, "2020-09-20", "A", 3.0), ("g2", 2020, "2020-09-20", "B", 0.0),
        ("g3", 2020, "2020-09-27", "A", 99.0), ("g3", 2020, "2020-09-27", "B", 0.0),
    ])
    pre = decayed_ratings(tg, half_life=1, offseason_weight=1.0).set_index(["game_id", "team"])
    assert np.isnan(pre.loc[("g1", "A"), "off_epa"])           # no history yet
    a_raw = (0.5 * 1 + 3) / 1.5                                 # g3's own 99 is not used
    league = (a_raw + 0.0) / 2
    assert pre.loc[("g3", "A"), "off_epa"] == pytest.approx(a_raw - league)
    assert pre.loc[("g3", "B"), "off_epa"] == pytest.approx(0.0 - league)


def test_feature_table_has_home_and_away_columns_per_group():
    tg = team_games([("g1", 2020, "2020-09-13", "A", 1.0), ("g1", 2020, "2020-09-13", "B", 0.0),
                     ("g2", 2020, "2020-09-20", "B", 0.0), ("g2", 2020, "2020-09-20", "A", 3.0)])
    games = pd.DataFrame({"game_id": ["g1", "g2"], "home_team": ["A", "B"], "away_team": ["B", "A"]})
    feats = team_stat_features(games, tg, half_life=8, offseason_weight=0.5, groups=["efficiency"])
    assert len(feats.columns) == 1 + 2 * len(GROUPS["efficiency"])
    assert feats.loc[1, "away_off_epa"] > 0 > feats.loc[1, "home_off_epa"]   # A was better in g1


def test_record_counts_wins_and_ties():
    games = pd.concat([one_game(), one_game().assign(game_id="g2", home_score=20, away_score=20, result=0)])
    pbp = pd.DataFrame([play("A", rush=1, epa=0.1), play("B", rush=1, epa=0.1),
                        play("A", game_id="g2", rush=1, epa=0.1), play("B", game_id="g2", rush=1, epa=0.1)])
    stats = team_game_stats(pbp, games).set_index(["game_id", "team"])
    assert stats.loc[("g1", "A"), "off_wins"] == 1 and stats.loc[("g1", "B"), "off_wins"] == 0
    assert stats.loc[("g2", "A"), "off_wins"] == 0.5
    assert GROUPS["record"] == ["off_wins"]
