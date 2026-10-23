"""Walk-forward season splits, the holdout guard, and evaluation metrics."""

from collections.abc import Iterable
from dataclasses import dataclass


class HoldoutError(RuntimeError):
    """Raised when code tries to use the locked holdout season."""


@dataclass(frozen=True)
class Split:
    test_season: int
    val_season: int                  # tuning, calibration, thresholds
    train_seasons: tuple[int, ...]   # every season from the first data season through test_season - 2


def check_holdout(seasons: Iterable[int], cfg: dict) -> None:
    """Raise HoldoutError if the holdout season is requested while it is locked."""
    ev = cfg["evaluation"]
    if ev["holdout_locked"] and ev["holdout_season"] in set(seasons):
        raise HoldoutError(
            f"season {ev['holdout_season']} is the locked holdout; "
            "set evaluation.holdout_locked to false only once the model and strategies are frozen"
        )


def walk_forward_splits(cfg: dict) -> list[Split]:
    """One split per test season: train through S-2, validate on S-1, test on S."""
    ev = cfg["evaluation"]
    first = cfg["data"]["first_season"]
    splits = [
        Split(test_season=s, val_season=s - 1, train_seasons=tuple(range(first, s - 1)))
        for s in range(ev["first_test_season"], ev["last_test_season"] + 1)
    ]
    check_holdout((s.test_season for s in splits), cfg)
    return splits
