import copy

import pytest

from nflmodel.evaluate import HoldoutError, check_holdout, walk_forward_splits


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
