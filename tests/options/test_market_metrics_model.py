"""Option Market Metrics Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.options import black_scholes_model, market_metrics_model

# pylint: disable=missing-function-docstring

STOCK_PRICE = 100.0
TIME = 30 / 365
RATE = 0.03


def chain(
    put_option: bool, volatility: float = 0.2, interest: float = 100.0
) -> pd.DataFrame:
    """Options priced with Black-Scholes at one volatility, quoted a cent around the price."""
    strikes = np.arange(50.0, 151.0, 1.0)
    prices = np.array(
        [
            black_scholes_model.get_black_scholes(
                stock_price=STOCK_PRICE,
                strike_price=float(strike),
                risk_free_rate=RATE,
                volatility=volatility,
                time_to_expiration=TIME,
                put_option=put_option,
            )
            for strike in strikes
        ]
    )
    return pd.DataFrame(
        {
            "Bid": np.maximum(prices - 0.01, 0.0),
            "Ask": prices + 0.01,
            "Last Price": prices,
            "Volume": 10.0,
            "Open Interest": interest,
            "Implied Volatility": volatility,
        },
        index=pd.Index(strikes, name="Strike Price"),
    )


def test_put_call_ratio():
    result = market_metrics_model.get_put_call_ratio(
        chain(False), chain(True, interest=200.0)
    )
    assert result["Volume Put-Call Ratio"] == 1
    assert result["Open Interest Put-Call Ratio"] == 2


def test_max_pain_lies_where_open_interest_is_balanced():
    calls = chain(False, interest=0.0)
    puts = chain(True, interest=0.0)
    calls.loc[90.0, "Open Interest"] = 1000
    puts.loc[110.0, "Open Interest"] = 1000
    # Between 90 and 110 neither side pays; the first such strike is reported.
    assert 90 <= market_metrics_model.get_max_pain(calls, puts) <= 110

    calls.loc[90.0, "Open Interest"] = 0
    assert market_metrics_model.get_max_pain(calls, puts) == 110


def test_gamma_exposure_signs_and_flip():
    exposure = market_metrics_model.get_gamma_exposure(
        chain(False), chain(True), STOCK_PRICE, TIME, RATE
    )
    assert (exposure["Call Gamma Exposure"] >= 0).all()
    assert (exposure["Put Gamma Exposure"] <= 0).all()
    # Equal open interest on both sides leaves almost no net exposure at the money.
    assert (
        abs(exposure["Net Gamma Exposure"].sum())
        < exposure["Call Gamma Exposure"].sum() * 0.05
    )

    heavy_puts = market_metrics_model.get_gamma_flip(
        [(chain(False), chain(True, interest=300.0), TIME)], STOCK_PRICE, RATE
    )
    assert np.isnan(heavy_puts) or heavy_puts > 0


def test_flat_volatility_gives_no_skew_and_the_right_level():
    skew = market_metrics_model.get_volatility_skew(
        chain(False), chain(True), STOCK_PRICE, TIME, RATE
    )
    assert skew["At-the-Money Volatility"] == pytest.approx(0.2)
    assert skew["25-Delta Risk Reversal"] == pytest.approx(0, abs=1e-9)
    assert skew["25-Delta Butterfly"] == pytest.approx(0, abs=1e-9)


def test_implied_move_is_the_straddle():
    move = market_metrics_model.get_implied_move(chain(False), chain(True), STOCK_PRICE)
    assert move["Straddle Strike"] == 100
    # An at-the-money straddle is worth about 0.8 standard deviations of the move.
    assert move["Implied Move"] == pytest.approx(0.8 * 0.2 * np.sqrt(TIME), rel=0.05)


def test_model_free_variance_recovers_a_flat_volatility():
    variance, forward = market_metrics_model.get_model_free_variance(
        chain(False), chain(True), TIME, RATE
    )
    assert np.sqrt(variance) == pytest.approx(0.2, abs=0.005)
    assert forward == pytest.approx(STOCK_PRICE * np.exp(RATE * TIME), rel=1e-3)

    assert market_metrics_model.get_constant_maturity_volatility(
        0.04, 20, 0.04, 40, 30
    ) == pytest.approx(0.2)


def test_mid_prices_fall_back_to_the_last_price():
    quotes = pd.DataFrame(
        {"Bid": [1.0, 0.0], "Ask": [1.2, 0.0], "Last Price": [1.05, 2.0]}, index=[1, 2]
    )
    assert market_metrics_model.get_mid_prices(quotes).tolist() == [1.1, 2.0]


def test_a_term_without_bids_has_no_variance():
    # Outside market hours the bids can be zero: only K0 is left, whose correction term
    # alone is a small negative variance, so the term has none.
    calls, puts = chain(False), chain(True)
    calls["Bid"], puts["Bid"] = 0.0, 0.0
    variance, _ = market_metrics_model.get_model_free_variance(calls, puts, TIME, RATE)

    assert np.isnan(variance)
    # One valid term, weighted, is not a constant maturity volatility.
    assert np.isnan(
        market_metrics_model.get_constant_maturity_volatility(
            variance, 20, 0.04, 40, 30
        )
    )
    assert np.isnan(
        market_metrics_model.get_constant_maturity_volatility(-1e-5, 20, 0.04, 40, 30)
    )
