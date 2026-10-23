import numpy as np
import pytest

from nflmodel.betting import SPREAD_PRICE, american_to_decimal, devig, implied_prob, moneyline_profit


@pytest.mark.parametrize(
    "odds, decimal",
    [(-110, 1 + 100 / 110), (+150, 2.5), (-150, 1 + 100 / 150), (+100, 2.0), (-100, 2.0), (-200, 1.5)],
)
def test_american_to_decimal(odds, decimal):
    assert american_to_decimal(odds) == pytest.approx(decimal)


def test_decimal_conversion_is_vectorized_and_keeps_nan():
    out = american_to_decimal([-110, np.nan, 200])
    assert out[0] == pytest.approx(1.9090909)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(3.0)


def test_odds_between_minus_and_plus_100_rejected():
    with pytest.raises(ValueError):
        american_to_decimal(50)


def test_standard_spread_price_needs_52_4_percent_to_break_even():
    assert implied_prob(SPREAD_PRICE) == pytest.approx(110 / 210)   # 0.5238


def test_devig_even_game():
    p_home, p_away = devig(implied_prob(-110), implied_prob(-110))
    assert p_home == pytest.approx(0.5)
    assert p_away == pytest.approx(0.5)


def test_devig_favorite_by_hand():
    # -200 implies 2/3, +170 implies 100/270; together 1.037, i.e. a 3.7% overround.
    raw_home, raw_away = implied_prob(-200), implied_prob(+170)
    assert raw_home + raw_away == pytest.approx(2 / 3 + 100 / 270)
    p_home, p_away = devig(raw_home, raw_away)
    assert p_home == pytest.approx((2 / 3) / (2 / 3 + 100 / 270))   # 0.6429
    assert p_home + p_away == pytest.approx(1.0)


def test_devig_rejects_unknown_method():
    with pytest.raises(ValueError):
        devig(0.6, 0.45, method="shin")


def test_moneyline_profit_win_loss_and_tie_refund():
    profit = moneyline_profit([-110, -110, +150, +150], [1, 0, 1, np.nan])
    np.testing.assert_allclose(profit, [100 / 110, -1.0, 1.5, 0.0])
