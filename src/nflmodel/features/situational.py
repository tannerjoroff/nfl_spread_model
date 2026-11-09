"""Situational features, all known from the schedule before kickoff."""

import pandas as pd

SHORT_WEEK_DAYS = 5      # rest of 5 days or fewer: Thursday games and similar
BYE_DAYS = 13            # rest of 13 days or more: coming off a bye

COLUMNS = ["neutral", "home_rest", "away_rest", "rest_diff", "home_short_week", "away_short_week",
           "home_off_bye", "away_off_bye", "div_game", "primetime"]


def situational_features(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game. Primetime means a start at 7pm ET or later, or any non-Sunday game."""
    out = pd.DataFrame({"game_id": games["game_id"]})
    out["neutral"] = (games["location"] == "Neutral").astype(float)
    for side in ("home", "away"):
        rest = games[f"{side}_rest"].astype(float)
        out[f"{side}_rest"] = rest
        out[f"{side}_short_week"] = (rest <= SHORT_WEEK_DAYS).astype(float)
        out[f"{side}_off_bye"] = (rest >= BYE_DAYS).astype(float)
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["div_game"] = games["div_game"].astype(float)
    late = games["gametime"].fillna("13:00") >= "19:00"
    out["primetime"] = (late | (games["weekday"] != "Sunday")).astype(float)
    return out[["game_id", *COLUMNS]]
