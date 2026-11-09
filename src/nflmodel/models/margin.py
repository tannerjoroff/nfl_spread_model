"""Margin models: predict the home team's margin from pre-game team ratings.

For each walk-forward split, every setting (decay half-life, offseason weight,
ridge penalty, XGBoost depth and tree count) is chosen on the last few seasons
before the test season, each scored by a model trained only on seasons before it.
The chosen model is refit on every season before the test season and predicts
the test season once.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from nflmodel.evaluate import Split, check_holdout
from nflmodel.features import situational
from nflmodel.features.team_stats import GROUPS, team_stat_features

# Groups that do not depend on decay settings: Elo (games must carry run_elo's columns)
# and the situational flags from the schedule.
ELO_COLUMNS = ["elo_home_pre", "elo_away_pre", "elo_margin"]
STATIC_GROUPS = {"elo": ELO_COLUMNS, "situational": situational.COLUMNS}


def _rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def feature_columns(groups: list[str]) -> list[str]:
    cols = []
    for g in groups:
        if g in STATIC_GROUPS:
            cols += STATIC_GROUPS[g]
        else:
            cols += [f"{side}_{c}" for side in ("home", "away") for c in GROUPS[g]]
    return cols


def feature_tables(games: pd.DataFrame, team_games: pd.DataFrame, cfg: dict) -> dict[tuple, pd.DataFrame]:
    """Features for every game under every decay setting in the config grid, aligned to games' rows."""
    grid = cfg["features"]["team_stats"]
    static = situational.situational_features(games).drop(columns="game_id")
    if set(ELO_COLUMNS) <= set(games.columns):
        static = static.join(games[ELO_COLUMNS])
    static.index = games.index
    tables = {}
    for h in grid["half_life_grid"]:
        for f in grid["offseason_weight_grid"]:
            X = team_stat_features(games, team_games, h, f).drop(columns="game_id")
            X.index = games.index
            tables[(h, f)] = X.join(static)
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


def validation_folds(split: Split, cfg: dict) -> list[tuple[tuple[int, ...], int]]:
    """(training seasons, validation season) for each of the last k seasons before the test season.
    Each validation season is scored by a model trained only on seasons before it."""
    k = cfg["evaluation"]["validation_seasons"]
    first = cfg["evaluation"]["first_train_season"]
    return [(tuple(range(first, v)), v) for v in range(split.test_season - k, split.test_season)]


def run_split(split: Split, games: pd.DataFrame, feats: dict[tuple, pd.DataFrame], cfg: dict,
              groups: list[str], seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """Tune on the validation folds, refit on every season before the test season, predict it once.

    feats maps (half_life, offseason_weight) -> feature table for every game, computed once:
    each game's features use only earlier games, so slicing by season later is safe.
    """
    check_holdout([split.test_season], cfg)
    first = cfg["evaluation"]["first_train_season"]
    season = games["season"]
    folds = [(season.isin(tr).to_numpy(), (season == v).to_numpy()) for tr, v in validation_folds(split, cfg)]
    fit_rows = season.between(first, split.test_season - 1).to_numpy()
    test = (season == split.test_season).to_numpy()
    y = games["result"].to_numpy(dtype=float)
    cols = feature_columns(groups)
    val_n = int(sum(va.sum() for _, va in folds))

    def ridge_val_rmse(X: pd.DataFrame, alpha: float) -> float:
        """Pooled RMSE over the validation folds: total squared error / total games."""
        sq = sum(((y[va] - fit_ridge(X.loc[tr, cols], y[tr], alpha).predict(X.loc[va, cols])) ** 2).sum()
                 for tr, va in folds)
        return float(np.sqrt(sq / val_n))

    # 1. Ridge chooses the decay settings and its own penalty, scored on the validation folds only.
    val_rmse, setting, alpha = min(
        (ridge_val_rmse(X, a), s, a) for s, X in feats.items() for a in cfg["models"]["ridge_alphas"]
    )
    X = feats[setting][cols]
    if X[fit_rows | test].isna().any().any():
        raise ValueError("missing features in training or test rows")

    preds = {"ridge": fit_ridge(X[fit_rows], y[fit_rows], alpha).predict(X[test])}
    choices = {"half_life": setting[0], "offseason_weight": setting[1], "ridge_alpha": alpha,
               "ridge_val_rmse": val_rmse, "val_games": val_n}

    # 2. XGBoost reuses those settings. Per depth: early stopping on each fold, RMSE pooled over
    #    folds, tree count = median over folds. Chosen per seed.
    for seed in seeds:
        tuned = []
        for d in cfg["models"]["xgb"]["max_depth_grid"]:
            fold_results = [(_xgb_val(X[tr], y[tr], X[va], y[va], d, seed, cfg), va.sum()) for tr, va in folds]
            pooled = np.sqrt(sum(r ** 2 * n for (r, _), n in fold_results) / val_n)
            n_trees = int(np.median([trees for (_, trees), _ in fold_results]))
            tuned.append((float(pooled), n_trees, d))
        xgb_val_rmse, n_trees, depth = min(tuned)
        preds[f"xgb_seed{seed}"] = fit_xgb(X[fit_rows], y[fit_rows], depth, n_trees, seed, cfg).predict(X[test])
        choices[f"xgb_seed{seed}"] = {"max_depth": depth, "n_trees": n_trees, "val_rmse": xgb_val_rmse}

    out = games.loc[test, ["game_id", "season", "week", "result"]].assign(**preds)
    return out.reset_index(drop=True), choices
