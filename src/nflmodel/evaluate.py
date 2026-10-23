"""Walk-forward season splits, the holdout guard, and evaluation metrics."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

PROB_CLIP = 1e-15   # keeps log loss finite when a probability is exactly 0 or 1


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


def margin_metrics(y_true, y_pred) -> dict:
    """MAE and RMSE of predicted home margins. Rows with a missing value are skipped."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    err = y_pred[ok] - y_true[ok]
    return {"mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean()), "n": int(ok.sum())}


def prob_metrics(y_true, p) -> dict:
    """Log loss, Brier score, and accuracy of home win probabilities.

    y_true is 1, 0, or NaN for a tie. Ties are dropped and counted in n_ties.
    """
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    if np.isnan(p).any():
        raise ValueError("probabilities contain NaN")
    ties = np.isnan(y_true)
    y, p = y_true[~ties], p[~ties]
    pc = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return {
        "log_loss": -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
        "brier": np.mean((p - y) ** 2),
        "accuracy": np.mean((p > 0.5) == (y == 1)),
        "n": int(len(y)),
        "n_ties": int(ties.sum()),
    }
