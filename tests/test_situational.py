import pandas as pd

from nflmodel.features.situational import COLUMNS, situational_features


def test_situational_by_hand():
    games = pd.DataFrame({
        "game_id": ["thu", "sun", "london"],
        "location": ["Home", "Home", "Neutral"],
        "home_rest": [4, 7, 14], "away_rest": [4, 14, 7],
        "div_game": [1, 0, 0],
        "gametime": ["20:15", "13:00", "09:30"],
        "weekday": ["Thursday", "Sunday", "Sunday"],
        "result": [3, -10, 7],
    })
    f = situational_features(games).set_index("game_id")
    assert list(f.columns) == COLUMNS
    assert f.loc["thu", ["home_short_week", "away_short_week", "primetime", "div_game"]].tolist() == [1, 1, 1, 1]
    assert f.loc["sun", ["away_off_bye", "home_off_bye", "primetime", "rest_diff"]].tolist() == [1, 0, 0, -7]
    assert f.loc["london", ["neutral", "home_off_bye", "primetime"]].tolist() == [1, 1, 0]


def test_situational_ignores_the_result():
    games = pd.DataFrame({"game_id": ["g"], "location": ["Home"], "home_rest": [7], "away_rest": [7],
                          "div_game": [0], "gametime": ["13:00"], "weekday": ["Sunday"], "result": [3]})
    changed = games.assign(result=-40)
    pd.testing.assert_frame_equal(situational_features(games), situational_features(changed))
