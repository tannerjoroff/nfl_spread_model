import copy

import numpy as np
import pandas as pd
import pytest

from nflmodel.evaluate import HoldoutError, Split
from nflmodel.models.margin import feature_columns, run_split

GROUPS = ["efficiency"]
SPLIT = Split(test_season=2012, val_season=2011, train_seasons=tuple(range(2007, 2011)))


@pytest.fixture
def small_cfg(cfg):
    small = copy.deepcopy(cfg)
    small["models"]["ridge_alphas"] = [1, 100]
    small["models"]["xgb"].update(max_depth_grid=[2, 3], max_trees=60, early_stopping_rounds=10)
    return small


@pytest.fixture
def data():
    """Six seasons of fake games. Under setting (8, 0.5) the first column predicts the margin;
    under (4, 0.5) every column is noise."""
    rng = np.random.default_rng(0)
    seasons = np.repeat(np.arange(2007, 2013), 200)
    cols = feature_columns(GROUPS)
    signal = rng.normal(0, 1, len(seasons))
    games = pd.DataFrame({"game_id": [f"g{i}" for i in range(len(seasons))], "season": seasons,
                          "week": np.tile(np.arange(1, 201) % 17 + 1, 6),
                          "result": 7 * signal + rng.normal(0, 10, len(seasons))})
    good = pd.DataFrame(rng.normal(0, 1, (len(seasons), len(cols))), columns=cols)
    good[cols[0]] = signal
    noise = pd.DataFrame(rng.normal(0, 1, (len(seasons), len(cols))), columns=cols)
    return games, {(4, 0.5): noise, (8, 0.5): good}


def test_validation_picks_the_informative_setting(small_cfg, data):
    games, feats = data
    _, choices = run_split(SPLIT, games, feats, small_cfg, GROUPS, seeds=[11])
    assert (choices["half_life"], choices["offseason_weight"]) == (8, 0.5)


def test_predicts_only_the_test_season(small_cfg, data):
    games, feats = data
    preds, _ = run_split(SPLIT, games, feats, small_cfg, GROUPS, seeds=[11, 22])
    assert set(preds["season"]) == {2012} and len(preds) == 200
    assert {"ridge", "xgb_seed11", "xgb_seed22"} <= set(preds.columns)


def test_test_season_outcomes_never_influence_choices_or_predictions(small_cfg, data):
    games, feats = data
    preds, choices = run_split(SPLIT, games, feats, small_cfg, GROUPS, seeds=[11])

    corrupted = games.copy()
    corrupted.loc[corrupted["season"] == 2012, "result"] = 1000.0
    preds2, choices2 = run_split(SPLIT, corrupted, feats, small_cfg, GROUPS, seeds=[11])

    assert choices2 == choices
    pd.testing.assert_frame_equal(preds.drop(columns="result"), preds2.drop(columns="result"))


def test_validation_outcomes_do_influence_choices(small_cfg, data):
    # Sanity check that the test above can fail: validation results must matter.
    games, feats = data
    _, choices = run_split(SPLIT, games, feats, small_cfg, GROUPS, seeds=[11])
    corrupted = games.copy()
    corrupted.loc[corrupted["season"] == 2011, "result"] = 1000.0
    _, choices2 = run_split(SPLIT, corrupted, feats, small_cfg, GROUPS, seeds=[11])
    assert choices2["ridge_val_rmse"] != choices["ridge_val_rmse"]


def test_holdout_split_refused(small_cfg, data):
    games, feats = data
    with pytest.raises(HoldoutError):
        run_split(Split(2025, 2024, (2007,)), games, feats, small_cfg, GROUPS, seeds=[11])
