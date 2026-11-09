"""Aggregate play-by-play into one row per team per game.

    python -m scripts.build_features

Writes data/interim/team_games.parquet (gitignored). Decayed ratings are cheap to
compute from it, so each half-life and offseason setting is built on the fly later.
While the holdout is locked, games from the holdout season onward are left out.
"""

import argparse
from pathlib import Path

from nflmodel.config import load_config
from nflmodel.data import read_raw
from nflmodel.features.team_stats import PBP_COLUMNS, team_game_stats, team_games_path
from nflmodel.games import build_games


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    last = ev["holdout_season"] - 1 if ev["holdout_locked"] else cfg["data"]["last_season"]
    seasons = list(range(cfg["data"]["first_season"], last + 1))

    games = build_games(cfg)
    games = games[games["season"].isin(seasons)]
    pbp = read_raw("pbp", seasons=seasons, cfg=cfg, columns=PBP_COLUMNS)
    team_games = team_game_stats(pbp, games)

    out = team_games_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    team_games.to_parquet(out, index=False)
    print(f"wrote {len(team_games)} team-games ({seasons[0]}-{seasons[-1]}) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
