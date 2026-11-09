"""Walk-forward margin models (ridge and XGBoost) for each feature group set.

    python -m scripts.train_margin

Needs data/interim/team_games.parquet from scripts.build_features. Writes:
- results/margin_models.csv: per test season and pooled MAE and RMSE for every model
  (XGBoost as the mean over seeds, with the range), next to the market and Elo.
- results/margin_validation.csv: pooled validation RMSE per group set, the number
  used to decide whether a feature group is kept.
- results/margin_intervals.csv: paired bootstrap intervals, our error minus the market's and Elo's.
- results/margin_choices.csv: the settings each split chose.
- data/interim/margin_predictions.parquet: test-season predictions for later phases.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nflmodel.config import load_config, repo_root
from nflmodel.data import data_dir
from nflmodel.evaluate import bootstrap_ci, check_holdout, walk_forward_splits
from nflmodel.features.team_stats import team_games_path
from nflmodel.games import build_games, select_game_types
from nflmodel.models.elo import run_elo
from nflmodel.models.margin import feature_tables, run_split

GROUP_SETS = {"efficiency": ["efficiency"], "efficiency+box_score": ["efficiency", "box_score"]}
RESULTS = Path("results")


def _errors(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df["result"]


def season_metrics(preds: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    rows = []
    xgb_cols = [f"xgb_seed{s}" for s in seeds]
    for label, g in [*((str(s), g) for s, g in preds.groupby("season")),
                     (f"{preds['season'].min()}-{preds['season'].max()}", preds)]:
        for model in ("market", "elo", "ridge"):
            e = _errors(g, model)
            rows.append({"model": model, "season": label, "n": len(g),
                         "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean())})
        maes = [_errors(g, c).abs().mean() for c in xgb_cols]
        rmses = [np.sqrt((_errors(g, c) ** 2).mean()) for c in xgb_cols]
        rows.append({"model": "xgb", "season": label, "n": len(g), "mae": np.mean(maes), "rmse": np.mean(rmses),
                     "mae_min": min(maes), "mae_max": max(maes), "rmse_min": min(rmses), "rmse_max": max(rmses)})
    return pd.DataFrame(rows)


def paired_intervals(preds: pd.DataFrame, seeds: list[int]) -> list[dict]:
    """Bootstrap our per-game error minus a baseline's, resampling weeks. Negative = we are better."""
    weeks = (preds["season"] * 100 + preds["week"]).to_numpy()
    rows = []
    # Ridge has one prediction, bootstrapped with each seed; XGBoost pairs each seed's model with its seed.
    for model, cols in (("ridge", ["ridge"] * len(seeds)), ("xgb", [f"xgb_seed{s}" for s in seeds])):
        for baseline in ("market", "elo"):
            for kind, f in (("abs_error", np.abs), ("squared_error", np.square)):
                cis = []
                for col, seed in zip(cols, seeds):
                    diff = (f(_errors(preds, col)) - f(_errors(preds, baseline))).to_numpy()
                    cis.append(bootstrap_ci(diff, clusters=weeks, seed=seed))
                rows.append({"model": model, "vs": baseline, "error": kind,
                             "estimate": np.mean([c["estimate"] for c in cis]),
                             "lo": np.mean([c["lo"] for c in cis]), "hi": np.mean([c["hi"] for c in cis]),
                             "se": np.mean([c["se"] for c in cis]), "n": cis[0]["n"], "weeks": cis[0]["n_clusters"]})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    splits = walk_forward_splits(cfg)
    last = max(s.test_season for s in splits)
    check_holdout([s.test_season for s in splits], cfg)
    seeds = cfg["seeds"]

    path = team_games_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} not found; run python -m scripts.build_features first")
    team_games = pd.read_parquet(path)

    all_games = build_games(cfg)
    all_games = all_games[all_games["season"] <= last]
    games = select_game_types(all_games, cfg)
    games = games[games["season"] >= cfg["evaluation"]["first_train_season"]].reset_index(drop=True)
    games = games.merge(run_elo(all_games)[["game_id", "elo_margin"]], on="game_id", how="left")
    games["market"], games["elo"] = games["spread_line"], games["elo_margin"]

    print("building features for every decay setting...")
    feats = feature_tables(games, team_games, cfg)

    metrics, intervals, choices, validation, all_preds = [], [], [], [], []
    for name, groups in GROUP_SETS.items():
        preds, val_sq, val_n = [], {"ridge": 0.0, "xgb": 0.0}, 0
        for split in splits:
            print(f"{name}: test {split.test_season}")
            p, ch = run_split(split, games, feats, cfg, groups, seeds)
            p = p.merge(games[["game_id", "market", "elo"]], on="game_id")
            preds.append(p)
            n_val = int((games["season"] == split.val_season).sum())
            val_n += n_val
            val_sq["ridge"] += n_val * ch["ridge_val_rmse"] ** 2
            val_sq["xgb"] += n_val * np.mean([ch[f"xgb_seed{s}"]["val_rmse"] ** 2 for s in seeds])
            flat = {k: v for k, v in ch.items() if not k.startswith("xgb_seed")}
            for s in seeds:
                flat[f"xgb_depth_seed{s}"] = ch[f"xgb_seed{s}"]["max_depth"]
                flat[f"xgb_trees_seed{s}"] = ch[f"xgb_seed{s}"]["n_trees"]
            choices.append({"groups": name, "test_season": split.test_season, **flat})
        preds = pd.concat(preds, ignore_index=True)
        all_preds.append(preds.assign(groups=name))
        metrics.append(season_metrics(preds, seeds).assign(groups=name))
        intervals.extend({"groups": name, **r} for r in paired_intervals(preds, seeds))
        validation.extend({"groups": name, "model": m, "pooled_val_rmse": np.sqrt(v / val_n), "val_games": val_n}
                          for m, v in val_sq.items())

    out = repo_root() / RESULTS
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "margin_models.csv": pd.concat(metrics, ignore_index=True),
        "margin_validation.csv": pd.DataFrame(validation),
        "margin_intervals.csv": pd.DataFrame(intervals),
        "margin_choices.csv": pd.DataFrame(choices),
    }
    for fname, df in tables.items():
        df.insert(0, "groups", df.pop("groups"))
        df.to_csv(out / fname, index=False, float_format="%.4f", encoding="utf-8", lineterminator="\n")
    pred_path = data_dir(cfg) / "interim" / "margin_predictions.parquet"
    pd.concat(all_preds, ignore_index=True).to_parquet(pred_path, index=False)

    pooled = tables["margin_models.csv"]
    pooled = pooled[pooled["season"].str.contains("-")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pooled.round(4).to_string(index=False))
        print()
        print(tables["margin_validation.csv"].round(4).to_string(index=False))
        print()
        print(tables["margin_intervals.csv"].round(4).to_string(index=False))
    print(f"\nwrote {', '.join(tables)} to results/ and predictions to {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
