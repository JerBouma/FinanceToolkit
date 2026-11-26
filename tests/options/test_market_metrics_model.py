"""Option Market Metrics Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.options.market import market_metrics_model
from financetoolkit.options.pricing import black_scholes_model

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


def test_a_quoted_straddle_has_its_spread_and_no_last_trade_legs():
    move = market_metrics_model.get_implied_move(chain(False), chain(True), STOCK_PRICE)
    # Each leg is quoted a cent either side of its price, so the spreads sum to 0.04.
    assert move["Bid-Ask Spread"] == pytest.approx(0.04 / move["Straddle Price"])
    assert move["Last Trade Legs"] == 0


def test_a_leg_without_a_quote_counts_as_a_last_trade_leg():
    calls = chain(False).assign(**{"Last Trade Date": "2026-11-09"})
    calls.loc[100.0, ["Bid", "Ask", "Last Price", "Open Interest"]] = [0, 1.75, 3.15, 1]

    move = market_metrics_model.get_implied_move(calls, chain(True), STOCK_PRICE)
    legs = market_metrics_model.get_straddle_quotes(calls, chain(True), 100.0)

    assert move["Last Trade Legs"] == 1
    assert np.isnan(move["Bid-Ask Spread"])
    assert move["Straddle Price"] == pytest.approx(
        3.15 + chain(True).loc[100.0, "Last Price"]
    )
    assert legs.loc["Call", "Last Trade Date"] == "2026-11-09"
    assert legs.loc["Call", "Open Interest"] == 1
    assert legs.loc["Put", "Bid-Ask Spread"] == pytest.approx(0.02)


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


def test_the_time_to_expiration_runs_to_four_in_the_afternoon_in_new_york():
    from financetoolkit.options import helpers

    # 30 hours before the close of the next day.
    now = pd.Timestamp("2026-11-17 10:00")
    assert helpers.get_time_to_expiration("2026-11-18", now) == pytest.approx(
        30 * 60 / 525_600
    )
    assert helpers.get_days_to_expiration("2026-11-18", now) == 1
    # A time in another time zone is converted: 15:00 UTC is 10:00 in New York.
    utc = pd.Timestamp("2026-11-17 15:00", tz="UTC")
    assert helpers.get_time_to_expiration("2026-11-18", utc) == pytest.approx(
        30 * 60 / 525_600
    )
    # After the close the option has expired.
    assert helpers.get_time_to_expiration("2026-11-17", now + pd.Timedelta(hours=7)) < 0
    assert helpers.get_days_to_expiration("2026-11-17", now) == 0


def test_a_time_that_daylight_saving_skips_or_repeats_is_accepted():
    from financetoolkit.options import helpers

    skipped = pd.Timestamp("2026-03-08 02:30")
    repeated = pd.Timestamp("2026-11-01 01:30")

    assert helpers.get_time_to_expiration("2026-03-09", skipped) > 0
    assert helpers.get_time_to_expiration("2026-11-02", repeated) > 0


def test_a_strike_without_a_quote_or_activity_has_no_price():
    # 160 still has open interest and 165 traded today; 170 has neither, so its last
    # price is stale and is not used (an implied volatility of 150% from it).
    quotes = pd.DataFrame(
        {
            "Bid": [2.4, 0.0, 0.0, 0.0],
            "Ask": [2.6, 0.0, 0.0, 0.0],
            "Last Price": [2.45, 3.1, 3.1, 3.1],
            "Volume": [1500.0, 0.0, 4.0, 0.0],
            "Open Interest": [8000.0, 12.0, 0.0, 0.0],
        },
        index=[100.0, 160.0, 165.0, 170.0],
    )
    prices = market_metrics_model.get_mid_prices(quotes)

    assert prices.iloc[:3].tolist() == pytest.approx([2.5, 3.1, 3.1])
    assert np.isnan(prices.iloc[3])


def priced_strip(lower: float, upper: float, step: float, volatility=None):
    """
    Strikes and Black-Scholes call and put prices around a stock price of 100, at one
    volatility or at a volatility per log-moneyness ln(K / F) given by a function.
    """
    strikes = np.arange(lower, upper + step / 2, step)
    forward = STOCK_PRICE * np.exp(RATE * TIME)
    vols = (
        np.full(strikes.size, 0.2)
        if volatility is None
        else volatility(np.log(strikes / forward))
    )

    def prices(put_option: bool) -> np.ndarray:
        return np.array(
            [
                black_scholes_model.get_black_scholes(
                    stock_price=STOCK_PRICE,
                    strike_price=float(strike),
                    risk_free_rate=RATE,
                    volatility=float(vol),
                    time_to_expiration=TIME,
                    put_option=put_option,
                )
                for strike, vol in zip(strikes, vols, strict=True)
            ]
        )

    return strikes, prices(False), prices(True), forward


def test_risk_neutral_moments_of_black_scholes_approach_the_normal_distribution():
    # The log return of Black-Scholes is normal: skewness 0 and kurtosis 3, which the
    # strip approaches as it widens and densifies.
    errors = []
    for lower, upper, step in ((70, 130, 2.5), (50, 150, 1), (10, 190, 0.1)):
        strikes, calls, puts, forward = priced_strip(lower, upper, step)
        moments = market_metrics_model.get_risk_neutral_moments(
            strikes, calls, puts, forward, RATE, TIME
        )
        errors.append(abs(moments["Kurtosis"] - 3))
        skewness = market_metrics_model.get_cboe_skewness(
            strikes, calls, puts, forward, RATE, TIME
        )

    assert errors[0] > errors[1] > errors[2]
    assert moments["Kurtosis"] == pytest.approx(3, abs=1e-3)
    assert moments["Skewness"] == pytest.approx(0, abs=1e-4)
    assert moments["Volatility"] == pytest.approx(0.2, abs=1e-4)
    # E[ln(S_T / F)] = -σ²T / 2 for a lognormal price.
    assert moments["Expected Log Return"] == pytest.approx(
        -(0.2**2) / 2 * TIME, abs=1e-6
    )
    assert skewness == pytest.approx(0, abs=1e-4)


def test_an_equity_smile_has_a_negative_skewness_and_a_skew_index_above_100():
    from financetoolkit.options.volatility import svi_model

    # An SVI slice with the downward slope of equity smiles: puts carry the higher volatility.
    def smile(log_moneyness):
        variance = svi_model.get_svi_total_variance(
            log_moneyness, a=0.0005, b=0.02, rho=-0.7, m=0.0, sigma=0.05
        )
        return np.sqrt(variance / TIME)

    strikes, calls, puts, forward = priced_strip(30, 170, 0.5, smile)
    moments = market_metrics_model.get_risk_neutral_moments(
        strikes, calls, puts, forward, RATE, TIME
    )
    skewness = market_metrics_model.get_cboe_skewness(
        strikes, calls, puts, forward, RATE, TIME
    )

    assert moments["Skewness"] < -0.5
    assert moments["Kurtosis"] > 3
    # The Cboe expansion around the forward and the expansion around K_0 agree.
    assert skewness == pytest.approx(moments["Skewness"], abs=0.01)
    _, index = market_metrics_model.get_skew_index(skewness, 20, skewness, 40, 30)
    assert index > 105


def test_the_skew_index_interpolates_like_the_white_paper_example():
    # Cboe SKEW white paper: S = 0.730208333 (-2.19656) + 0.269791667 (-1.68) for the
    # August (near) and September (next) 2010 terms, 28 days apart. The paper prints -2.056
    # and 120.56 from the unrounded September skewness (about -1.676).
    next_days = 30 + 0.730208333 * 28
    skewness, index = market_metrics_model.get_skew_index(
        -2.19656, next_days - 28, -1.68, next_days, 30
    )

    assert skewness == pytest.approx(0.730208333 * -2.19656 + 0.269791667 * -1.68)
    assert index == pytest.approx(100 - 10 * skewness)
    assert index == pytest.approx(120.56, abs=0.02)
    assert np.isnan(market_metrics_model.get_skew_index(np.nan, 20, -2, 40)[1])


def test_out_of_the_money_prices_take_puts_below_and_calls_above_the_forward():
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    calls = np.array([11.0, 6.5, 3.0, 1.0, np.nan, 0.1])
    puts = np.array([0.2, 0.6, 2.0, 5.0, 9.5, 14.4])

    used, prices, spacing, k0 = market_metrics_model.get_out_of_the_money_prices(
        strikes, calls, puts, 101.0
    )

    # 110 has no call price, so 105 and 115 cover its interval.
    assert k0 == 100
    assert used.tolist() == [90, 95, 100, 105, 115]
    assert prices.tolist() == [0.2, 0.6, 2.5, 1.0, 0.1]
    assert spacing.tolist() == [5, 5, 5, 7.5, 10]
    assert (
        market_metrics_model.get_out_of_the_money_prices(strikes, calls, puts, 80.0)[
            0
        ].size
        == 0
    )


def test_the_moments_use_the_strikes_of_the_vix_strip():
    calls, puts = chain(False), chain(True)
    strip, forward = market_metrics_model.get_model_free_strip(calls, puts, TIME, RATE)
    # The far wings are worth less than the cent of the spread and have no bid: the strip
    # runs from the lowest bid put to the highest bid call.
    assert strip.index[0] == puts.index[puts["Bid"] > 0].min()
    assert strip.index[-1] == calls.index[calls["Bid"] > 0].max()
    assert strip.index.is_monotonic_increasing
    assert forward == pytest.approx(STOCK_PRICE * np.exp(RATE * TIME), rel=1e-3)

    calls, puts = chain(False), chain(True)
    calls["Bid"], puts["Bid"] = 0.0, 0.0
    strip, _ = market_metrics_model.get_model_free_strip(calls, puts, TIME, RATE)
    assert strip.empty


def test_the_cboe_skewness_does_not_depend_on_the_expansion_point():
    # Strikes 7 apart leave the forward (100.25) far above K_0 = 99: the ε terms carry the
    # expansion from K_0 to F_0, so both expansions of the same strip give one skewness.
    strikes, calls, puts, forward = priced_strip(1, 197, 7)
    moments = market_metrics_model.get_risk_neutral_moments(
        strikes, calls, puts, forward, RATE, TIME
    )
    skewness = market_metrics_model.get_cboe_skewness(
        strikes, calls, puts, forward, RATE, TIME
    )

    assert strikes[strikes <= forward].max() == 99
    assert skewness == pytest.approx(moments["Skewness"], abs=1e-4)


def black_scholes_strip(
    strikes, volatility, stock_price, rate, time, dividend_yield=0.0
):
    """Call and put prices at each strike, with a volatility per strike."""
    calls, puts = [], []
    for strike in strikes:
        for put_option, prices in ((False, calls), (True, puts)):
            prices.append(
                black_scholes_model.get_black_scholes(
                    stock_price=stock_price,
                    strike_price=float(strike),
                    risk_free_rate=rate,
                    volatility=float(volatility(strike)),
                    time_to_expiration=time,
                    dividend_yield=dividend_yield,
                    put_option=put_option,
                )
            )
    return np.array(calls), np.array(puts)


def test_the_variance_swap_fair_strike_matches_quantlib():
    # QuantLib 1.44 ReplicatingVarianceSwapEngine (strike step 5) on a skewed
    # BlackVarianceSurface, 91 days, spot 100, r = 5%: VarianceSwap.variance().
    time = 91 / 365
    strikes = np.arange(50.0, 151.0, 5.0)
    calls, puts = black_scholes_strip(
        strikes, lambda strike: 0.30 - 0.1 * (strike - 100) / 100, 100.0, 0.05, time
    )
    fair = market_metrics_model.get_variance_swap_fair_strike(
        strikes,
        calls,
        puts,
        100 * np.exp(0.05 * time),
        0.05,
        time,
        boundary_strike=100.0,
    )
    assert fair == pytest.approx(0.09166679082402854, rel=1e-12)


def test_the_variance_swap_fair_strike_matches_financepy():
    # FinancePy 1.1.2 EquityVarianceSwap.fair_strike: 18 strikes of 5 on each side of the
    # forward, a flat 25% volatility, r = 5%, no dividends, 90 days.
    time = 90 / 365
    forward = 100 * np.exp(0.05 * time)
    strikes = forward + 5.0 * np.arange(-17, 18)
    calls, puts = black_scholes_strip(strikes, lambda _: 0.25, 100.0, 0.05, time)
    fair = market_metrics_model.get_variance_swap_fair_strike(
        strikes, calls, puts, forward, 0.05, time, boundary_strike=forward
    )
    # FinancePy prices the options from its volatility curve with its own normal distribution,
    # which differs from scipy's in the seventh digit.
    assert fair == pytest.approx(0.06422763647544352, rel=1e-6)


def test_the_variance_swap_fair_strike_converges_to_the_variance():
    # Under Black-Scholes the log contract pays σ² T whatever the dividends, so finer strikes
    # bring the fair strike to σ². FinancePy 1.1.2 gives 0.0841 for this case with a 1%
    # dividend yield (and 0.0642 without), off by a fifth.
    time, rate, dividend_yield = 90 / 365, 0.05, 0.01
    forward = 100 * np.exp((rate - dividend_yield) * time)
    errors = []
    for step in (5.0, 1.0, 0.25):
        strikes = forward + step * np.arange(-int(95 / step) + 1, int(190 / step))
        calls, puts = black_scholes_strip(
            strikes, lambda _: 0.25, 100.0, rate, time, dividend_yield
        )
        fair = market_metrics_model.get_variance_swap_fair_strike(
            strikes, calls, puts, forward, rate, time, boundary_strike=forward
        )
        errors.append(fair - 0.0625)
    assert errors[0] > errors[1] > errors[2] > 0
    assert errors[2] < 1e-5


def test_the_variance_swap_and_vix_weights_agree_for_fine_strikes():
    time = 30 / 365
    strikes = np.arange(60.0, 160.0, 0.5)
    weights = market_metrics_model.get_variance_swap_weights(strikes[80:], 100.0, time)
    vix = (2 / time) * 0.5 / strikes[80:] ** 2
    np.testing.assert_allclose(weights[1:-1], vix[1:-1], rtol=1e-2)
    calls, puts = black_scholes_strip(
        strikes, lambda k: 0.2 + (100 - k) / 500, 100.0, RATE, time
    )
    forward = 100 * np.exp(RATE * time)
    fair = market_metrics_model.get_variance_swap_fair_strike(
        strikes, calls, puts, forward, RATE, time
    )
    strike_values, quotes, spacing, k0 = (
        market_metrics_model.get_out_of_the_money_prices(strikes, calls, puts, forward)
    )
    cboe = (2 / time) * np.sum(
        spacing / strike_values**2 * np.exp(RATE * time) * quotes
    ) - (forward / k0 - 1) ** 2 / time
    assert fair == pytest.approx(cboe, rel=2e-3)


def test_the_variance_swap_fair_strike_needs_both_sides():
    strikes = np.array([90.0, 100.0, 110.0])
    calls = np.array([np.nan, 2.0, 1.0])
    puts = np.array([1.0, np.nan, np.nan])
    assert np.isnan(
        market_metrics_model.get_variance_swap_fair_strike(
            strikes, calls, puts, 100.0, 0.0, 0.1
        )
    )


def test_realized_variance_has_no_mean_and_keeps_each_history():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 99.0, 100.0], "B": [np.nan, 50.0, 51.0, 50.5]}
    )
    result = market_metrics_model.get_realized_variance(prices)
    returns_a = np.diff(np.log([100.0, 101.0, 99.0, 100.0]))
    returns_b = np.diff(np.log([50.0, 51.0, 50.5]))
    assert result["A"] == pytest.approx(252 * np.mean(returns_a**2))
    assert result["B"] == pytest.approx(252 * np.mean(returns_b**2))


def test_the_variance_swap_value_matches_financepy():
    # FinancePy 1.1.2 EquityVarianceSwap.value a month into a 90-day swap struck at 0.06,
    # realized 0.05, fair 0.07, with the discount curve anchored at the valuation date.
    value = market_metrics_model.get_variance_swap_value(
        realized_variance=0.05,
        fair_variance=0.07,
        variance_strike=0.06,
        elapsed_fraction=31 / 90,
        remaining_time=59 / 365,
        risk_free_rate=0.05,
    )
    assert value == pytest.approx(0.0030860678529896113, rel=1e-12)


def test_a_new_variance_swap_at_its_fair_strike_is_worth_nothing():
    value = market_metrics_model.get_variance_swap_value(
        np.nan, 0.04, 0.04, 0.0, 0.5, 0.03, variance_notional=1e6
    )
    assert value == 0
    with pytest.raises(ValueError, match="between 0 and 1"):
        market_metrics_model.get_variance_swap_value(0.04, 0.04, 0.04, 1.5, 0.5, 0.03)


def test_a_vega_notional_converts_to_a_variance_notional():
    # A vega notional of 100,000 at a 20% strike is 100 * 100,000 / (2 * 0.2) = 25,000,000
    # of variance notional, which pays 102,500 when 21% is realized.
    notional = 100 * 100_000 / (2 * np.sqrt(0.2**2))
    assert market_metrics_model.get_variance_swap_value(
        0.21**2, 0.0, 0.2**2, 1.0, 0.0, 0.04, notional
    ) == pytest.approx(102_500)
