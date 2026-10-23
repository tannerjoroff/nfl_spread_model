"""Score the market baselines on every walk-forward test season.

    python -m scripts.baselines

Writes results/market_baselines.csv: one row per test season plus a pooled row.
"""

import argparse
from pathlib import Path

import pandas as pd

from nflmodel.config import load_config, repo_root
from nflmodel.evaluate import check_holdout, walk_forward_splits
from nflmodel.games import build_games, select_game_types
from nflmodel.models.baselines import score_market

OUTPUT = Path("results") / "market_baselines.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    test_seasons = [s.test_season for s in walk_forward_splits(cfg)]
    check_holdout(test_seasons, cfg)

    games = select_game_types(build_games(cfg), cfg)
    games = games[games["season"].isin(test_seasons)]

    rows = [{"season": str(s), **score_market(g)} for s, g in games.groupby("season")]
    rows.append({"season": f"{test_seasons[0]}-{test_seasons[-1]}", **score_market(games)})
    table = pd.DataFrame(rows)

    out = repo_root() / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", encoding="utf-8", lineterminator="\n")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    print(f"\nwrote {OUTPUT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
