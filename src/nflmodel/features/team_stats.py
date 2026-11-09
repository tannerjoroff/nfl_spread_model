"""Team ratings from play-by-play: decayed, pre-game, and relative to the league.

Three feature groups share this machinery:
- efficiency: EPA per play (all, pass, rush) and success rate.
- box_score: points per game, yards per play (all, pass, rush), explosive play rate,
  sack rate, turnover rate, third-down conversion rate, red zone touchdown rate,
  penalty yards per game.
- record: win percentage.

Each team-game row holds totals for the team's offense ("off_") and for the
opponent's offense in that game ("def_", what the defense allowed). For defense,
a higher number is usually worse. Ratings for a game use only games played on
earlier dates, so neither the game itself nor any game on the same day leaks in.
"""

import numpy as np
import pandas as pd

from nflmodel.games import FRANCHISE

PBP_COLUMNS = [
    "game_id", "posteam", "pass", "rush", "qb_kneel", "qb_spike", "epa", "success",
    "yards_gained", "sack", "interception", "fumble_lost", "third_down_converted",
    "third_down_failed", "yardline_100", "fixed_drive", "fixed_drive_result",
    "penalty", "penalty_team", "penalty_yards",
]

# stat -> the count it is divided by
EFFICIENCY = {"epa": "plays", "success": "plays", "pass_epa": "pass_plays", "rush_epa": "rush_plays"}
BOX_SCORE = {
    "points": "games",
    "yards": "plays",
    "pass_yards": "pass_plays",
    "rush_yards": "rush_plays",
    "explosive": "plays",          # passes of 20+ yards and runs of 10+
    "sacks": "pass_plays",
    "turnovers": "plays",          # interceptions and lost fumbles
    "third_conv": "third_att",
    "rz_td": "rz_trips",           # drives reaching the opponent's 20 that end in a touchdown
    "penalty_yards": "games",      # penalty yards committed
}
RECORD = {"wins": "games"}          # win percentage, a tie counting as half
RATES = {**EFFICIENCY, **BOX_SCORE, **RECORD}
GROUPS = {
    "efficiency": [f"{side}_{s}" for side in ("off", "def") for s in EFFICIENCY],
    "box_score": [f"{side}_{s}" for side in ("off", "def") for s in BOX_SCORE],
    "record": ["off_wins"],         # def_wins would just be the opponent's record in the same games
}

_FLAGS = ["pass", "rush", "qb_kneel", "qb_spike", "sack", "interception", "fumble_lost",
          "third_down_converted", "third_down_failed", "penalty", "penalty_yards"]


def team_game_stats(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per played game, with offensive totals (off_) and totals allowed (def_)."""
    pbp = pbp.copy()
    pbp[_FLAGS] = pbp[_FLAGS].fillna(0)

    plays = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["epa"].notna()
                & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]
    is_pass, is_rush, y = plays["pass"] == 1, plays["rush"] == 1, plays["yards_gained"].fillna(0)
    per_play = pd.DataFrame({
        "game_id": plays["game_id"], "team": plays["posteam"],
        "plays": 1.0,
        "pass_plays": is_pass.astype(float),
        "rush_plays": is_rush.astype(float),
        "epa": plays["epa"],
        "success": plays["success"].fillna(0),
        "pass_epa": plays["epa"].where(is_pass, 0.0),
        "rush_epa": plays["epa"].where(is_rush, 0.0),
        "yards": y,
        "pass_yards": y.where(is_pass, 0.0),
        "rush_yards": y.where(is_rush, 0.0),
        "explosive": ((is_pass & (y >= 20)) | (is_rush & (y >= 10))).astype(float),
        "sacks": plays["sack"],
        "turnovers": plays["interception"] + plays["fumble_lost"],
    })
    stats = per_play.groupby(["game_id", "team"]).sum()

    third = pbp.groupby(["game_id", "posteam"])[["third_down_converted", "third_down_failed"]].sum()
    third.index.names = ["game_id", "team"]
    stats["third_conv"] = third["third_down_converted"]
    stats["third_att"] = third["third_down_converted"] + third["third_down_failed"]

    rz = pbp[(pbp["yardline_100"] <= 20) & pbp["posteam"].notna() & pbp["fixed_drive"].notna()]
    rz = rz.drop_duplicates(["game_id", "posteam", "fixed_drive"])
    rz_by = rz.assign(td=rz["fixed_drive_result"] == "Touchdown").groupby(["game_id", "posteam"])["td"]
    stats["rz_trips"] = rz_by.size().rename_axis(["game_id", "team"])
    stats["rz_td"] = rz_by.sum().rename_axis(["game_id", "team"])

    pen = pbp[pbp["penalty"] == 1].groupby(["game_id", "penalty_team"])["penalty_yards"].sum()
    stats["penalty_yards"] = pen.rename_axis(["game_id", "team"])

    played = games[games["result"].notna()]
    home = _franchise(played["home_team"])
    away = _franchise(played["away_team"])
    base = pd.concat([
        pd.DataFrame({"game_id": played["game_id"], "season": played["season"], "gameday": played["gameday"],
                      "team": home, "opp": away, "points": played["home_score"],
                      "opp_points": played["away_score"]}),
        pd.DataFrame({"game_id": played["game_id"], "season": played["season"], "gameday": played["gameday"],
                      "team": away, "opp": home, "points": played["away_score"],
                      "opp_points": played["home_score"]}),
    ], ignore_index=True)

    stats = stats.reset_index()
    points = base[["game_id", "team", "points"]].assign(
        games=1.0, wins=(base["points"] > base["opp_points"]) + 0.5 * (base["points"] == base["opp_points"]))
    stats = points.merge(stats, on=["game_id", "team"], how="left").fillna(0)

    off = stats.add_prefix("off_").rename(columns={"off_game_id": "game_id", "off_team": "team"})
    dfn = stats.add_prefix("def_").rename(columns={"def_game_id": "game_id", "def_team": "opp"})
    out = base.drop(columns=["points", "opp_points"]).merge(off, on=["game_id", "team"]).merge(dfn, on=["game_id", "opp"])
    return out.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)


def _franchise(teams: pd.Series) -> np.ndarray:
    return teams.replace(FRANCHISE).to_numpy()


def _cumulative_rates(team_games: pd.DataFrame, half_life: float, offseason_weight: float) -> pd.DataFrame:
    """Each team's decayed rate including each of its games (its rating right *after* that game)."""
    tg = team_games.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    n = tg.groupby("team").cumcount()
    # Each newer game counts 2^(1/half_life) times the one before it, and each season
    # boundary multiplies everything older by offseason_weight. Only ratios matter,
    # so weights are scaled to a max of 1 per team to stay far from overflow.
    log_w = n * np.log(2) / half_life - tg["season"] * np.log(offseason_weight)
    w = np.exp(log_w - log_w.groupby(tg["team"]).transform("max"))

    out = tg[["team", "gameday"]].copy()
    for side in ("off", "def"):
        for stat, denom in RATES.items():
            num = (w * tg[f"{side}_{stat}"]).groupby(tg["team"]).cumsum()
            den = (w * tg[f"{side}_{denom}"]).groupby(tg["team"]).cumsum()
            out[f"{side}_{stat}"] = num / den.where(den > 0)
    return out


def decayed_ratings(team_games: pd.DataFrame, half_life: float, offseason_weight: float) -> pd.DataFrame:
    """Pre-game ratings for every team-game, relative to the league, from strictly earlier games."""
    cum = _cumulative_rates(team_games, half_life, offseason_weight).sort_values("gameday")
    cols = [c for c in cum.columns if c.startswith(("off_", "def_"))]

    def as_of(left: pd.DataFrame) -> pd.DataFrame:
        # Last rating from a game played strictly before each row's gameday.
        return pd.merge_asof(left.sort_values("gameday"), cum, on="gameday", by="team",
                             allow_exact_matches=False)

    # League baseline on each game date: average of every team's rating as of that date.
    dates = pd.DataFrame({"gameday": np.sort(team_games["gameday"].unique())})
    grid = dates.merge(pd.DataFrame({"team": team_games["team"].unique()}), how="cross")
    league = as_of(grid).groupby("gameday")[cols].mean()

    pre = as_of(team_games[["game_id", "team", "gameday"]])
    pre[cols] = pre[cols].to_numpy() - league.loc[pre["gameday"], cols].to_numpy()
    return pre[["game_id", "team", *cols]]


def team_stat_features(games: pd.DataFrame, team_games: pd.DataFrame, half_life: float,
                       offseason_weight: float, groups: list[str] | None = None) -> pd.DataFrame:
    """One row per game: home_ and away_ pre-game ratings for the chosen feature groups."""
    cols = [c for g in (groups or list(GROUPS)) for c in GROUPS[g]]
    pre = decayed_ratings(team_games, half_life, offseason_weight)[["game_id", "team", *cols]]
    out = games[["game_id"]].copy()
    for side in ("home", "away"):
        keys = pd.DataFrame({"game_id": games["game_id"], "team": _franchise(games[f"{side}_team"])})
        side_pre = keys.merge(pre, on=["game_id", "team"], how="left")
        out[[f"{side}_{c}" for c in cols]] = side_pre[cols].to_numpy()
    return out


def team_games_path(cfg: dict):
    """Where build_features saves the team-game table (gitignored, rebuilt from the snapshot)."""
    from nflmodel.data import data_dir
    return data_dir(cfg) / "interim" / "team_games.parquet"
