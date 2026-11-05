"""Factors Model Tests"""

# pylint: disable=missing-function-docstring

import numpy as np
import pandas as pd
import pytest

from financetoolkit.factors import factors_model

TICKERS = list("ABCDEFGHIJ")


def random_prices(periods=400, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, (periods, len(TICKERS)))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2020-01-01", periods=periods),
        columns=TICKERS,
    )


def test_a_factor_equal_to_the_forward_return_has_an_ic_of_one():
    prices = factors_model.sample_by_period(random_prices(), "monthly")
    forward_returns = factors_model.get_forward_returns(prices, 1)

    perfect = factors_model.get_information_coefficient(
        forward_returns, forward_returns
    )
    reversed_ = factors_model.get_information_coefficient(
        -forward_returns, forward_returns
    )

    assert np.allclose(perfect.dropna(), 1)
    assert np.allclose(reversed_.dropna(), -1)
    # The last month has no forward return, so no IC.
    assert np.isnan(perfect.iloc[-1])


def test_the_ic_is_the_spearman_correlation_of_the_tickers_with_both_values():
    factor = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, np.nan]], columns=list("ABCDE"))
    forward_returns = pd.DataFrame([[0.1, 0.3, 0.2, 0.4, 0.5]], columns=list("ABCDE"))

    ic = factors_model.get_information_coefficient(factor, forward_returns)

    # Ranks 1-2-3-4 against 1-3-2-4: Spearman's rho = 1 - 6 * 2 / (4 * 15) = 0.8.
    assert ic.iloc[0] == pytest.approx(0.8)


def test_a_date_with_fewer_than_three_tickers_has_no_ic():
    factor = pd.DataFrame([[1.0, 2.0, np.nan]], columns=list("ABC"))
    forward_returns = pd.DataFrame([[0.1, 0.2, 0.3]], columns=list("ABC"))

    assert np.isnan(
        factors_model.get_information_coefficient(factor, forward_returns).iloc[0]
    )


def test_forward_returns_skip_the_days_a_ticker_does_not_trade():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, np.nan, 120.0]},
        index=pd.period_range("2024-01-02", periods=3, freq="D"),
    )

    forward_returns = factors_model.get_forward_returns(prices, 1)

    assert forward_returns["A"].iloc[0] == pytest.approx(0.1)
    # B's next close after the first day is the third day's.
    assert forward_returns["B"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(forward_returns["B"].iloc[1])


def test_sampling_takes_the_last_value_of_each_period_and_leaves_unlisted_tickers_empty():
    data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0], "B": [np.nan, np.nan, 5.0, np.nan]},
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"]),
    )

    sampled = factors_model.sample_by_period(data, "monthly")

    assert sampled["A"].tolist() == [2.0, 4.0]
    assert np.isnan(sampled["B"].iloc[0])
    assert sampled["B"].iloc[1] == 5.0  # noqa: PLR2004


def test_quantiles_are_equally_sized_and_need_enough_tickers():
    factor = pd.DataFrame(
        [np.arange(10.0), [1.0, 2.0, 3.0] + [np.nan] * 7], columns=TICKERS
    )

    quantiles = factors_model.assign_quantiles(factor, 5)

    assert quantiles.iloc[0].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert quantiles.iloc[1].isna().all()


def test_a_perfect_factor_orders_the_quantile_returns():
    prices = factors_model.sample_by_period(random_prices(), "monthly")
    forward_returns = factors_model.get_forward_returns(prices, 1)
    quantiles = factors_model.assign_quantiles(forward_returns, 5)

    returns = factors_model.get_quantile_returns(quantiles, forward_returns)

    quantile_returns = returns.drop("Top Minus Bottom")
    assert quantile_returns.is_monotonic_increasing
    assert returns["Top Minus Bottom"] > 0
    # The spread is the average of the per-date differences, which with every quantile
    # present on every date equals the difference of the averages.
    assert returns["Top Minus Bottom"] == pytest.approx(
        returns["Quantile 5"] - returns["Quantile 1"]
    )


def test_an_unchanged_ranking_has_no_turnover_and_a_rank_autocorrelation_of_one():
    factor = pd.DataFrame([np.arange(10.0)] * 4, columns=TICKERS)
    quantiles = factors_model.assign_quantiles(factor, 5)

    turnover = factors_model.get_quantile_turnover(quantiles)
    autocorrelation = factors_model.get_factor_rank_autocorrelation(factor)

    assert turnover.iloc[0].isna().all()
    assert (turnover.iloc[1:] == 0).all().all()
    assert np.isnan(autocorrelation.iloc[0])
    assert np.allclose(autocorrelation.iloc[1:], 1)


def test_a_reversed_ranking_turns_every_outer_quantile_over():
    factor = pd.DataFrame([np.arange(10.0), -np.arange(10.0)], columns=TICKERS)
    quantiles = factors_model.assign_quantiles(factor, 5)

    turnover = factors_model.get_quantile_turnover(quantiles)

    assert turnover["Quantile 1"].iloc[1] == 1
    assert turnover["Quantile 5"].iloc[1] == 1
    assert factors_model.get_factor_rank_autocorrelation(factor).iloc[
        1
    ] == pytest.approx(-1)


def test_the_summary_follows_the_ic_per_date():
    ic = pd.DataFrame({"1 Month": [0.1, 0.3, -0.1, 0.2], "3 Months": [0.5] * 4})

    summary = factors_model.summarize_information_coefficient(ic)

    mean, deviation = ic["1 Month"].mean(), ic["1 Month"].std(ddof=1)
    assert summary.loc["Mean IC", "1 Month"] == pytest.approx(mean)
    assert summary.loc["IC Information Ratio", "1 Month"] == pytest.approx(
        mean / deviation
    )
    assert summary.loc["t-Statistic", "1 Month"] == pytest.approx(
        mean / (deviation / 2)
    )
    assert summary.loc["Positive Share", "1 Month"] == pytest.approx(0.75)
    assert summary.loc["Observations", "1 Month"] == 4  # noqa: PLR2004
    # A constant IC has no deviation and so no ratio.
    assert np.isnan(summary.loc["IC Information Ratio", "3 Months"])


def test_horizon_labels():
    assert factors_model.get_horizon_label(1, "monthly") == "1 Month"
    assert factors_model.get_horizon_label(21, "daily") == "21 Days"
