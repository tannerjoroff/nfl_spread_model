"""Market baselines: what the betting market already predicts for each game.

These need no training. Any model of ours has to beat them to be worth betting.
"""

import numpy as np
import pandas as pd

from nflmodel.betting import devig, implied_prob, moneyline_profit
from nflmodel.evaluate import margin_metrics, prob_metrics


def market_margin(games: pd.DataFrame) -> pd.Series:
    """Predicted home margin = the spread (nflverse spread_line is already the home margin)."""
    return games["spread_line"].astype(float)


def market_win_prob(games: pd.DataFrame) -> pd.Series:
    """De-vigged home win probability from the two moneylines. NaN where a moneyline is missing."""
    p_home, _ = devig(implied_prob(games["home_moneyline"]), implied_prob(games["away_moneyline"]))
    return pd.Series(p_home, index=games.index, name="p_home")


def favorite_bets(games: pd.DataFrame) -> pd.DataFrame:
    """One 1-unit moneyline bet on the market favorite of every game with both moneylines.

    Pick'em games (de-vigged 50/50) have no favorite and are skipped. won is 1, 0,
    or NaN for a tie, whose stake is refunded (profit 0).
    """
    p = market_win_prob(games)
    bets = games[p.notna() & (p != 0.5)]
    home_fav = (p[bets.index] > 0.5).to_numpy()
    odds = np.where(home_fav, bets["home_moneyline"], bets["away_moneyline"])
    won = np.where(home_fav, bets["home_win"], 1 - bets["home_win"])
    return pd.DataFrame({
        "game_id": bets["game_id"].to_numpy(),
        "season": bets["season"].to_numpy(),
        "week": bets["week"].to_numpy(),
        "won": won,
        "profit": moneyline_profit(odds, won),
    })


def score_market(games: pd.DataFrame) -> dict:
    """Score the market baselines on a set of games.

    Win probability metrics use games with both moneylines (missing ones are counted).
    "Bet every favorite" stakes 1 unit on the moneyline favorite of every game; a tie
    refunds the stake and is left out of ROI. fav_roi_se treats bets as independent;
    see bootstrap_ci for intervals that resample weeks.
    """
    margins = margin_metrics(games["result"], market_margin(games))

    p_home = market_win_prob(games)
    has_ml = p_home.notna()
    probs = prob_metrics(games.loc[has_ml, "home_win"], p_home[has_ml])

    bets = favorite_bets(games)
    graded = bets[bets["won"].notna()]

    return {
        "games": len(games),
        "ties": int(games["home_win"].isna().sum()),
        "margin_mae": margins["mae"],
        "margin_rmse": margins["rmse"],
        "ml_missing": int((~has_ml).sum()),
        "log_loss": probs["log_loss"],
        "brier": probs["brier"],
        "accuracy": probs["accuracy"],
        "fav_win_rate": (graded["won"] == 1).mean(),
        "fav_bets": len(graded),
        "fav_roi": graded["profit"].mean(),
        "fav_roi_se": graded["profit"].std(ddof=1) / np.sqrt(len(graded)),
    }
