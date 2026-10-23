"""Odds conversion, removing the bookmaker's margin (vig), and bet profit.

American odds: -150 means stake 150 to win 100; +150 means stake 100 to win 150.
Decimal odds: total returned per 1 staked, including the stake (-110 -> 1.909).
"""

import numpy as np

SPREAD_PRICE = -110   # every spread bet is priced at a flat -110


def american_to_decimal(odds):
    """Convert American odds to decimal odds. NaN stays NaN."""
    odds = np.asarray(odds, dtype=float)
    if np.any(np.abs(odds[~np.isnan(odds)]) < 100):
        raise ValueError("American odds must be <= -100 or >= +100")
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / -odds)


def implied_prob(odds):
    """Break-even win probability of a price. The two sides of a game sum to more than 1."""
    return 1 / american_to_decimal(odds)


def devig(p_home_raw, p_away_raw, method: str = "multiplicative"):
    """Remove the vig so the two probabilities sum to 1.

    Multiplicative: divide each implied probability by their sum, which shrinks
    both sides in proportion to their size.
    """
    if method != "multiplicative":
        raise ValueError(f"unknown de-vig method {method!r}")
    p_home_raw = np.asarray(p_home_raw, dtype=float)
    p_away_raw = np.asarray(p_away_raw, dtype=float)
    total = p_home_raw + p_away_raw
    return p_home_raw / total, p_away_raw / total


def moneyline_profit(odds, won):
    """Profit in units on a 1-unit bet. won is 1 (win), 0 (loss), or NaN (tie: stake refunded)."""
    payout = american_to_decimal(odds) - 1
    won = np.asarray(won, dtype=float)
    return np.where(np.isnan(won), 0.0, np.where(won == 1, payout, -1.0))
