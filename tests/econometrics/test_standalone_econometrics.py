"""Econometrics on any data (Econometrics(data=...)) and the regime, beta, correlation and pairs methods."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Econometrics


@pytest.fixture(scope="module")
def monthly_rates():
    rng = np.random.default_rng(0)
    index = pd.period_range("2000-01", periods=240, freq="M")
    first = np.cumsum(rng.normal(0, 0.001, 240)) + 0.03
    second = 0.5 * first + rng.normal(0, 0.0005, 240) + 0.01
    return pd.DataFrame({"United States": first, "Germany": second}, index=index)


def test_standalone_econometrics_on_any_data(monthly_rates):
    econometrics = Econometrics(data=monthly_rates, changes="difference")

    adf = econometrics.get_augmented_dickey_fuller()
    assert list(adf.columns) == ["United States", "Germany"]
    # The series are analyzed at the dates they are observed: 240 months.
    assert adf.loc["Observations"].max() <= 240  # noqa: PLR2004

    ols = econometrics.get_ols(dependent_ticker="Germany", column="Level")
    assert ols.loc["United States", "Coefficient"] == pytest.approx(0.5, abs=0.05)

    yearly = econometrics.get_augmented_dickey_fuller(period="yearly", column="Change")
    assert yearly.loc["Observations"].max() < 20  # noqa: PLR2004


def test_standalone_needs_data():
    with pytest.raises(ValueError, match="data"):
        Econometrics()
    with pytest.raises(ValueError, match="either data"):
        Econometrics(
            data=pd.Series([1.0], index=pd.to_datetime(["2026-01-01"])),
            historical_data={},
        )


def test_new_methods_on_the_toolkit(econometrics_module):
    regimes = econometrics_module.get_markov_regime_switching(period="weekly")
    assert list(regimes.columns) == ["AAPL", "MSFT"]
    assert (
        regimes.loc["Regime 1 Volatility"].lt(regimes.loc["Regime 2 Volatility"]).all()
    )

    probabilities = econometrics_module.get_markov_regime_switching(
        period="weekly", probabilities=True
    )
    assert probabilities["AAPL"].sum(axis=1).to_numpy() == pytest.approx(1)
    assert list(probabilities.columns.get_level_values(1).unique()) == [
        "Regime 1",
        "Regime 2",
    ]

    breaks = econometrics_module.get_structural_breaks(period="weekly", max_breaks=2)
    assert breaks.index.names == ["Asset", "Segment"]

    beta = econometrics_module.get_time_varying_beta(period="weekly")
    assert set(beta.columns.get_level_values(0)) == {"AAPL", "MSFT"}
    assert beta[("AAPL", "Beta")].iloc[-1] == pytest.approx(1.2, abs=0.8)

    correlations = econometrics_module.get_dcc_garch()
    assert list(correlations.columns) == ["AAPL / MSFT"]
    assert correlations["AAPL / MSFT"].between(-1, 1).all()
    parameters = econometrics_module.get_dcc_garch(parameters=True)
    assert "DCC Alpha" in parameters.index

    pairs = econometrics_module.get_pairs_trading(window=60)
    assert pairs.index.names == ["Dependent", "Independent"]
    assert {"Hedge Ratio", "Half-Life", "Cointegration P-Value"} <= set(pairs.columns)
    series = econometrics_module.get_pairs_trading(time_series=True)
    assert list(series.columns.get_level_values(1).unique()) == [
        "Spread",
        "Z-Score",
        "Signal",
    ]


def test_standalone_mixed_monthly_and_daily_series():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-02", "2025-12-31")
    daily = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), days, name="Daily"
    )
    months = pd.date_range("2020-01-31", periods=72, freq="ME")
    monthly = pd.Series(
        100 + np.cumsum(rng.normal(0, 1, 72)) + np.r_[np.zeros(36), np.full(36, 15.0)],
        months,
        name="Monthly",
    )
    econometrics = Econometrics(data={"Daily": daily, "Monthly": monthly})

    # Every series is tested on its own observations, not only on the dates both share.
    jarque_bera = econometrics.get_jarque_bera_test()
    assert jarque_bera.loc["P-Value"].notna().all()
    adf = econometrics.get_augmented_dickey_fuller(column="Level", max_lag=0)
    assert adf.loc["Observations", "Daily"] == len(days) - 1
    assert adf.loc["Observations", "Monthly"] == 71  # noqa: PLR2004

    breaks = econometrics.get_structural_breaks(column="Level", breaks=1)
    assert breaks.loc[("Monthly", 1), "Observations"] == pytest.approx(36, abs=3)
    assert breaks.loc["Daily", "Observations"].sum() == len(days)

    regimes = econometrics.get_markov_regime_switching(probabilities=True)
    assert regimes["Daily"].dropna().shape[0] == len(days) - 1
    assert regimes["Monthly"].dropna().shape[0] == 71  # noqa: PLR2004


def test_standalone_arima_uses_every_series_at_its_own_observations():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2021-01-01", periods=750)
    daily = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), days, name="Price"
    )
    monthly = pd.Series(
        np.linspace(2, 4, 36),
        pd.date_range("2021-01-31", periods=36, freq="ME"),
        name="CPI",
    )
    econometrics = Econometrics(data={"Price": daily, "CPI": monthly})

    forecast = econometrics.get_arima_forecast(p=1, d=1, q=0, forecast_steps=2)
    # On only the month ends both share, the daily price would forecast from its last
    # month-end value; on its own observations it starts from its last daily value.
    assert forecast["Price"].iloc[0] == pytest.approx(daily.iloc[-1], rel=0.01)
    assert forecast["CPI"].notna().all()

    validation = econometrics.get_out_of_sample_validation(
        model="arima", p=1, d=1, q=0, train_fraction=0.8
    )
    assert validation.loc["Holdout Observations", "Price"] == len(days) - int(
        len(days) * 0.8
    )
    assert validation.loc["Holdout Observations", "CPI"] == 36 - int(36 * 0.8)


def test_level_significance_on_any_data(monthly_rates):
    econometrics = Econometrics(data=monthly_rates, rounding=None)

    significance = econometrics.get_level_significance(period="monthly")
    assert list(significance.index) == ["United States", "Germany"]

    for country in significance.index:
        history = monthly_rates[country]
        latest = history.iloc[-1]
        row = significance.loc[country]
        assert row["Latest Value"] == pytest.approx(latest)
        assert row["Start Date"] == "2000-01"
        assert row["Percentile"] == pytest.approx((history.iloc[:-1] < latest).mean())
        # The dates are the months of the data, the last at least as high and as low.
        for column, as_far in (
            ("Highest Since", history >= latest),
            ("Lowest Since", history <= latest),
        ):
            earlier = as_far.iloc[:-1]
            expected = str(earlier[earlier].index[-1]) if earlier.any() else None
            assert (row[column] if isinstance(row[column], str) else None) == expected

    # At the dates they are observed (the default), the months are their last days.
    daily = econometrics.get_level_significance()
    assert daily.loc["Germany", "Start Date"] == "2000-01-31"


def test_level_significance_on_the_toolkit(econometrics_module):
    significance = econometrics_module.get_level_significance()

    # The traded close of every ticker, the Benchmark only when asked for.
    assert "Benchmark" not in significance.index
    assert significance["Percentile"].between(0, 1).all()
    assert (
        "Benchmark"
        in econometrics_module.get_level_significance(include_benchmark=True).index
    )


def test_spreads_and_streaks_on_any_data():
    months = pd.period_range("2021-05", periods=6, freq="M")
    prices = pd.DataFrame(
        {
            "Producer Price Index YoY": [0.09, 0.088, -0.013, -0.007, 0.005, 0.057],
            "Inflation Rate YoY": [0.013, 0.011, 0.021, 0.018, 0.010, 0.007],
        },
        index=months,
    )
    econometrics = Econometrics(data=prices, rounding=None)

    # The spread is named after both series and measured like any series; the name of the
    # subtracted series is matched without regard to case.
    spread = econometrics.get_level_significance(
        period="monthly", spread_over="inflation rate yoy"
    )
    assert list(spread.index) == ["Producer Price Index YoY - Inflation Rate YoY"]
    assert spread.iloc[0]["Latest Value"] == pytest.approx(0.05)
    assert spread.iloc[0]["Highest Since"] == "2021-06"

    streaks = econometrics.get_streaks(period="monthly")
    assert streaks.loc["Producer Price Index YoY", "Run Start"] == "2021-09"
    assert streaks.loc["Producer Price Index YoY", "Previous Run Length"] == 2
    assert streaks.loc["Inflation Rate YoY", "Run Length"] == 6  # noqa: PLR2004

    spread_streaks = econometrics.get_streaks(
        period="monthly", spread_over="Inflation Rate YoY"
    )
    assert spread_streaks.iloc[0]["Run Start"] == "2021-10"
    assert spread_streaks.iloc[0]["Previous Run Start"] == "2021-07"

    # Changes count rises and falls: the producer prices rose in the last two months.
    rises = Econometrics(data=prices, changes="difference").get_streaks(
        period="monthly", column="Return"
    )
    assert rises.loc["Producer Price Index YoY", "Run Length"] == 3  # noqa: PLR2004

    # An unknown series or a spread without a second series is explained.
    assert econometrics.get_streaks(spread_over="Unemployment").empty
    assert (
        Econometrics(data=prices[["Inflation Rate YoY"]])
        .get_level_significance(spread_over="Inflation Rate YoY")
        .empty
    )
