"""Margin models: predict the home team's margin from pre-game team ratings.

For each walk-forward split, every setting (decay half-life, offseason weight,
ridge penalty, XGBoost depth and tree count) is chosen on the validation season.
The chosen model is refit on training plus validation and predicts the test
season once.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from nflmodel.evaluate import Split, check_holdout
from nflmodel.features.team_stats import GROUPS, team_stat_features


def _rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def feature_columns(groups: list[str]) -> list[str]:
    return [f"{side}_{c}" for side in ("home", "away") for g in groups for c in GROUPS[g]]


def feature_tables(games: pd.DataFrame, team_games: pd.DataFrame, cfg: dict) -> dict[tuple, pd.DataFrame]:
    """Features for every game under every decay setting in the config grid, aligned to games' rows."""
    grid = cfg["features"]["team_stats"]
    tables = {}
    for h in grid["half_life_grid"]:
        for f in grid["offseason_weight_grid"]:
            X = team_stat_features(games, team_games, h, f).drop(columns="game_id")
            X.index = games.index
            tables[(h, f)] = X
    return tables


def fit_ridge(X, y, alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y)


def _xgb(depth: int, n_trees: int, seed: int, cfg: dict, **extra) -> XGBRegressor:
    x = cfg["models"]["xgb"]
    return XGBRegressor(n_estimators=n_trees, max_depth=depth, learning_rate=x["learning_rate"],
                        subsample=x["subsample"], colsample_bytree=x["colsample_bytree"],
                        random_state=seed, n_jobs=1, **extra)


def _xgb_val(X_tr, y_tr, X_val, y_val, depth: int, seed: int, cfg: dict) -> tuple[float, int]:
    """Validation RMSE and tree count, with early stopping on the validation season."""
    x = cfg["models"]["xgb"]
    model = _xgb(depth, x["max_trees"], seed, cfg, early_stopping_rounds=x["early_stopping_rounds"],
                 eval_metric="rmse")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    n_trees = model.best_iteration + 1
    return _rmse(y_val, model.predict(X_val, iteration_range=(0, n_trees))), n_trees


def fit_xgb(X, y, depth: int, n_trees: int, seed: int, cfg: dict) -> XGBRegressor:
    return _xgb(depth, n_trees, seed, cfg).fit(X, y)


def run_split(split: Split, games: pd.DataFrame, feats: dict[tuple, pd.DataFrame], cfg: dict,
              groups: list[str], seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """Tune on the validation season, refit on train + validation, predict the test season once.

    feats maps (half_life, offseason_weight) -> feature table for every game, computed once:
    each game's features use only earlier games, so slicing by season later is safe.
    """
    check_holdout([split.test_season], cfg)
    train = games["season"].isin(split.train_seasons).to_numpy()
    val = (games["season"] == split.val_season).to_numpy()
    test = (games["season"] == split.test_season).to_numpy()
    fit_rows = train | val
    y = games["result"].to_numpy(dtype=float)
    cols = feature_columns(groups)

    # 1. Ridge chooses the decay settings and its own penalty, scored on validation only.
    val_rmse, setting, alpha = min(
        (_rmse(y[val], fit_ridge(X.loc[train, cols], y[train], a).predict(X.loc[val, cols])), s, a)
        for s, X in feats.items() for a in cfg["models"]["ridge_alphas"]
    )
    X = feats[setting][cols]
    if X[fit_rows | test].isna().any().any():
        raise ValueError("missing features in training or test rows")

    preds = {"ridge": fit_ridge(X[fit_rows], y[fit_rows], alpha).predict(X[test])}
    choices = {"half_life": setting[0], "offseason_weight": setting[1], "ridge_alpha": alpha,
               "ridge_val_rmse": val_rmse}

    # 2. XGBoost reuses those settings; depth and tree count are chosen on validation per seed.
    for seed in seeds:
        (xgb_val_rmse, n_trees), depth = min(
            (_xgb_val(X[train], y[train], X[val], y[val], d, seed, cfg), d)
            for d in cfg["models"]["xgb"]["max_depth_grid"]
        )
        preds[f"xgb_seed{seed}"] = fit_xgb(X[fit_rows], y[fit_rows], depth, n_trees, seed, cfg).predict(X[test])
        choices[f"xgb_seed{seed}"] = {"max_depth": depth, "n_trees": n_trees, "val_rmse": xgb_val_rmse}

    out = games.loc[test, ["game_id", "season", "week", "result"]].assign(**preds)
    return out.reset_index(drop=True), choices
