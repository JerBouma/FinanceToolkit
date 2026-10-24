"""Portfolio Attribution Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio import attribution_model

# pylint: disable=missing-function-docstring


def test_brinson_effects_add_up_to_the_excess_return():
    result = attribution_model.get_brinson_fachler_attribution(
        portfolio_weights=pd.Series({"Tech": 0.6, "Health": 0.4}),
        portfolio_returns=pd.Series({"Tech": 0.12, "Health": 0.03}),
        benchmark_weights=pd.Series({"Tech": 0.5, "Health": 0.3, "Energy": 0.2}),
        benchmark_returns=pd.Series({"Tech": 0.10, "Health": 0.05, "Energy": -0.02}),
    )
    total = result.loc["Total"]

    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"]
    )
    # Not holding Energy, which lagged, is a pure allocation effect.
    assert result.loc["Energy", "Selection Effect"] == 0
    assert result.loc["Energy", "Allocation Effect"] == pytest.approx(
        -0.2 * (-0.02 - total["Benchmark Return"])
    )


def test_money_weighted_return():
    one_year = pd.Series(
        [-100, 110], index=pd.to_datetime(["2020-01-01", "2020-12-31"])
    )
    assert attribution_model.get_money_weighted_return(one_year) == pytest.approx(0.1)

    with pytest.raises(ValueError, match="money invested and money received"):
        attribution_model.get_money_weighted_return(one_year.abs())


def test_stress_test_scales_up_the_assets_with_data():
    prices = pd.DataFrame(
        {"A": [100, 80, 90], "B": [np.nan, 50, 55]},
        index=pd.period_range("2020-02-18", periods=3, freq="D"),
    )
    returns = attribution_model.get_scenario_returns(
        prices, {"S": ("2020-02-18", "2020-02-20"), "T": ("2020-02-19", "2020-02-20")}
    )
    stress = attribution_model.get_stress_test(returns, pd.Series({"A": 0.5, "B": 0.5}))

    assert stress.loc["S", "Coverage"] == 0.5
    assert stress.loc["S", "Portfolio Return"] == pytest.approx(-0.1)
    assert stress.loc["T", "Portfolio Return"] == pytest.approx(0.1125)


def test_look_through_exposure():
    sectors = attribution_model.get_look_through_exposure(
        pd.Series({"QQQ": 0.5, "AAPL": 0.5}),
        {
            "QQQ": pd.Series({"Technology": 0.6, "Health": 0.4}),
            "AAPL": pd.Series({"Technology": 1.0}),
        },
    )
    assert sectors["Technology"] == pytest.approx(0.8)

    holdings = attribution_model.get_look_through_exposure(
        pd.Series({"QQQ": 0.5, "AAPL": 0.5}),
        {
            "QQQ": pd.Series({"NVDA": 0.1, "AAPL": 0.1}),
            "AAPL": pd.Series({"AAPL": 1.0}),
        },
    )
    assert holdings["AAPL"] == pytest.approx(0.55)
    assert holdings["Other"] == pytest.approx(0.4)
