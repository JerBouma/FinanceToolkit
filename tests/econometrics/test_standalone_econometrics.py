"""Econometrics on any data (Econometrics(data=...)) and the regime, beta, correlation and pairs methods."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Econometrics

# pylint: disable=missing-function-docstring


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
