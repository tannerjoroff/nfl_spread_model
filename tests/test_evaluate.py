import copy
import math

import numpy as np
import pytest

from nflmodel.evaluate import (
    HoldoutError,
    check_holdout,
    margin_metrics,
    prob_metrics,
    walk_forward_splits,
)


def test_one_split_per_test_season(cfg):
    splits = walk_forward_splits(cfg)
    assert [s.test_season for s in splits] == list(range(2012, 2025))


def test_train_before_validation_before_test(cfg):
    for s in walk_forward_splits(cfg):
        assert s.val_season == s.test_season - 1
        assert max(s.train_seasons) == s.test_season - 2
        assert s.val_season not in s.train_seasons


def test_training_window_expands_from_first_season(cfg):
    splits = walk_forward_splits(cfg)
    assert all(s.train_seasons[0] == 2006 for s in splits)
    assert splits[0].train_seasons == tuple(range(2006, 2011))
    assert len(splits[-1].train_seasons) == len(splits[0].train_seasons) + 12


def test_holdout_never_in_any_split(cfg):
    for s in walk_forward_splits(cfg):
        assert 2025 not in {s.test_season, s.val_season, *s.train_seasons}


def test_guard_blocks_holdout_while_locked(cfg):
    check_holdout([2023, 2024], cfg)
    with pytest.raises(HoldoutError):
        check_holdout([2024, 2025], cfg)


def test_guard_allows_holdout_once_unlocked(cfg):
    unlocked = copy.deepcopy(cfg)
    unlocked["evaluation"]["holdout_locked"] = False
    check_holdout([2025], unlocked)


def test_splits_refuse_a_config_that_tests_on_the_holdout(cfg):
    bad = copy.deepcopy(cfg)
    bad["evaluation"]["last_test_season"] = 2025
    with pytest.raises(HoldoutError):
        walk_forward_splits(bad)


# ---- metrics, checked against values computed by hand


def test_margin_metrics_by_hand():
    # errors are -2 and +6: MAE = 4, RMSE = sqrt((4 + 36) / 2) = sqrt(20)
    m = margin_metrics([3, -7], [1, -1])
    assert m["mae"] == pytest.approx(4.0)
    assert m["rmse"] == pytest.approx(math.sqrt(20))
    assert m["n"] == 2


def test_margin_metrics_skip_missing():
    assert margin_metrics([3, np.nan], [1, 0])["n"] == 1


def test_prob_metrics_by_hand():
    # log loss = -(ln 0.8 + ln 0.7) / 2; Brier = (0.2^2 + 0.3^2) / 2
    m = prob_metrics([1, 0], [0.8, 0.3])
    assert m["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert m["brier"] == pytest.approx(0.065)
    assert m["accuracy"] == 1.0


def test_coin_flip_reference_values():
    # Always saying 50% scores ln(2) = 0.693 log loss and 0.25 Brier: the bar any model must clear.
    m = prob_metrics([1, 0, 1, 1], [0.5] * 4)
    assert m["log_loss"] == pytest.approx(math.log(2))
    assert m["brier"] == pytest.approx(0.25)


def test_ties_dropped_and_counted():
    m = prob_metrics([1, np.nan, 0], [0.8, 0.5, 0.3])
    assert m["n"] == 2 and m["n_ties"] == 1
    assert m["brier"] == pytest.approx(0.065)


def test_confident_wrong_prediction_stays_finite():
    assert np.isfinite(prob_metrics([0], [1.0])["log_loss"])


def test_missing_probability_rejected():
    with pytest.raises(ValueError):
        prob_metrics([1], [np.nan])
