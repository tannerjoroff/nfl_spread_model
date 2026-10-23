import numpy as np
import pandas as pd
import pytest

from nflmodel.models.baselines import market_margin, market_win_prob, score_market


@pytest.fixture
def games():
    # g1: home favorite wins.  g2: away favorite wins.  g3: pick'em, no favorite to bet.
    # g4: home favorite ties (refund).  g5: moneylines missing.
    return pd.DataFrame({
        "result":         [3,     -7,    1,     0,     10],
        "home_win":       [1.0,   0.0,   1.0,   np.nan, 1.0],
        "spread_line":    [4.0,   -3.5,  0.0,   7.0,   2.5],
        "home_moneyline": [-200,  150,   -110,  -300,  np.nan],
        "away_moneyline": [170,   -180,  -110,  240,   np.nan],
    })


def test_market_margin_is_the_spread(games):
    np.testing.assert_array_equal(market_margin(games), games["spread_line"])


def test_market_win_prob_by_hand(games):
    p = market_win_prob(games)
    assert p[0] == pytest.approx((2 / 3) / (2 / 3 + 100 / 270))
    assert p[2] == pytest.approx(0.5)
    assert np.isnan(p[4])


def test_score_market_by_hand(games):
    s = score_market(games)
    assert s["games"] == 5
    assert s["ties"] == 1
    assert s["ml_missing"] == 1
    # Graded favorite bets: g1 at -200 (+0.5) and g2 at -180 (+100/180). g3 has no favorite, g4 is refunded.
    assert s["fav_bets"] == 2
    assert s["fav_win_rate"] == 1.0
    assert s["fav_roi"] == pytest.approx((0.5 + 100 / 180) / 2)
    # margin errors: 1, 3.5, -1, 7, -7.5
    assert s["margin_mae"] == pytest.approx((1 + 3.5 + 1 + 7 + 7.5) / 5)
