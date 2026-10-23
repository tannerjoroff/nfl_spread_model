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


def score_market(games: pd.DataFrame) -> dict:
    """Score the market baselines on a set of games.

    Win probability metrics use games with both moneylines (missing ones are counted).
    "Bet every favorite" stakes 1 unit on the moneyline favorite of every game; a tie
    refunds the stake and is left out of ROI. fav_roi_se treats bets as independent
    (bootstrap intervals come later).
    """
    margins = margin_metrics(games["result"], market_margin(games))

    p_home = market_win_prob(games)
    has_ml = p_home.notna()
    ml = games[has_ml]
    probs = prob_metrics(ml["home_win"], p_home[has_ml])

    p = p_home[has_ml]
    fav = ml[p != 0.5]
    home_fav = p[p != 0.5] > 0.5
    fav_odds = np.where(home_fav, fav["home_moneyline"], fav["away_moneyline"])
    fav_won = np.where(home_fav, fav["home_win"], 1 - fav["home_win"])
    profit = moneyline_profit(fav_odds, fav_won)
    decided = ~np.isnan(fav_won)
    graded = profit[decided]

    return {
        "games": len(games),
        "ties": int(games["home_win"].isna().sum()),
        "margin_mae": margins["mae"],
        "margin_rmse": margins["rmse"],
        "ml_missing": int((~has_ml).sum()),
        "log_loss": probs["log_loss"],
        "brier": probs["brier"],
        "fav_win_rate": (fav_won[decided] == 1).mean(),
        "fav_bets": len(graded),
        "fav_roi": graded.mean(),
        "fav_roi_se": graded.std(ddof=1) / np.sqrt(len(graded)),
    }
