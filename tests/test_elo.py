import math

import pandas as pd
import pytest

from nflmodel.models.elo import EloParams, mov_multiplier, run_elo, win_prob


def make_games(rows):
    """rows: (season, gameday, home, away, result) or with a trailing location."""
    out = []
    for i, r in enumerate(rows):
        season, day, home, away, result, *loc = r
        out.append({"game_id": f"g{i}", "season": season, "gameday": pd.Timestamp(day),
                    "home_team": home, "away_team": away, "result": result,
                    "location": loc[0] if loc else "Home"})
    return pd.DataFrame(out)


def test_win_prob_reference_points():
    assert win_prob(0) == pytest.approx(0.5)
    assert win_prob(400) == pytest.approx(10 / 11)
    assert win_prob(-400) == pytest.approx(1 / 11)


def test_first_game_uses_only_home_field():
    elo = run_elo(make_games([(2020, "2020-09-13", "A", "B", 7)]))
    assert elo.loc[0, "elo_prob_home"] == pytest.approx(win_prob(48))   # about 0.569
    assert elo.loc[0, "elo_margin"] == pytest.approx(48 / 25)


def test_neutral_site_has_no_home_field():
    elo = run_elo(make_games([(2020, "2020-09-13", "A", "B", 7, "Neutral")]))
    assert elo.loc[0, "elo_prob_home"] == pytest.approx(0.5)


def test_update_by_hand_and_zero_sum():
    games = make_games([
        (2020, "2020-09-13", "A", "B", 7),
        (2020, "2020-09-20", "A", "B", 3),
    ])
    elo = run_elo(games)
    p = win_prob(48)
    shift = 20 * mov_multiplier(7, 48) * (1 - p)
    assert elo.loc[1, "elo_home_pre"] == pytest.approx(1500 + shift)
    assert elo.loc[1, "elo_away_pre"] == pytest.approx(1500 - shift)


def test_upset_moves_ratings_more_than_expected_win():
    win = run_elo(make_games([(2020, "2020-09-13", "A", "B", 7), (2020, "2020-09-20", "A", "B", 0)]))
    loss = run_elo(make_games([(2020, "2020-09-13", "A", "B", -7), (2020, "2020-09-20", "A", "B", 0)]))
    gain = win.loc[1, "elo_home_pre"] - 1500
    drop = 1500 - loss.loc[1, "elo_home_pre"]
    assert drop > gain > 0


def test_bigger_margin_bigger_move_but_flattening():
    m3, m14, m28 = mov_multiplier(3, 0), mov_multiplier(14, 0), mov_multiplier(28, 0)
    assert m3 < m14 < m28
    assert (m28 - m14) < (m14 - m3)


def test_favorite_blowout_rewarded_less():
    assert mov_multiplier(21, winner_gap=200) < mov_multiplier(21, winner_gap=0)


def test_tie_between_equal_teams_at_neutral_site_changes_nothing():
    elo = run_elo(make_games([
        (2020, "2020-09-13", "A", "B", 0, "Neutral"),
        (2020, "2020-09-20", "A", "B", 3, "Neutral"),
    ]))
    assert elo.loc[1, "elo_home_pre"] == pytest.approx(1500)


def test_regression_between_seasons():
    games = make_games([
        (2020, "2020-09-13", "A", "B", 30),
        (2021, "2021-09-12", "A", "B", 3),
    ])
    elo = run_elo(games)
    end_2020 = 1500 + 20 * mov_multiplier(30, 48) * (1 - win_prob(48))
    assert elo.loc[1, "elo_home_pre"] == pytest.approx(1500 + (2 / 3) * (end_2020 - 1500))


def test_relocated_franchise_keeps_its_rating():
    elo = run_elo(make_games([
        (2015, "2015-09-13", "STL", "B", 14),
        (2015, "2015-09-20", "LA", "B", 3),
    ]))
    assert elo.loc[1, "elo_home_pre"] > 1500


def test_unplayed_game_gets_prediction_without_update():
    elo = run_elo(make_games([
        (2020, "2020-09-13", "A", "B", math.nan),
        (2020, "2020-09-20", "A", "B", 3),
    ]))
    assert elo.loc[1, "elo_home_pre"] == pytest.approx(1500)


def test_custom_params_used():
    elo = run_elo(make_games([(2020, "2020-09-13", "A", "B", 7)]), EloParams(home_adv=0))
    assert elo.loc[0, "elo_prob_home"] == pytest.approx(0.5)
