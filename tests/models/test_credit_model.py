"""Credit Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from financetoolkit.models.distress import credit_model


@pytest.mark.parametrize("time_horizon", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("default_point", [0.0, 80.0])
def test_merton_model_rejects_invalid_horizons_before_solving(
    time_horizon, default_point, mocker
):
    solver = mocker.patch("financetoolkit.models.distress.credit_model.fsolve")

    result = credit_model.get_merton_model(
        100.0, 0.4, default_point, 0.04, time_horizon=time_horizon
    )

    solver.assert_not_called()
    assert result["Default Point"] == default_point
    assert result.drop("Default Point").isna().all()


def test_merton_model_solves_both_equations():
    result = credit_model.get_merton_model(100, 0.4, 80, 0.04)
    value, volatility = result["Asset Value"], result["Asset Volatility"]
    d1 = (np.log(value / 80) + (0.04 + volatility**2 / 2)) / volatility
    d2 = d1 - volatility

    assert value * norm.cdf(d1) - 80 * np.exp(-0.04) * norm.cdf(d2) == pytest.approx(
        100
    )
    assert norm.cdf(d1) * volatility * value == pytest.approx(0.4 * 100)
    assert result["Probability of Default"] == pytest.approx(
        norm.cdf(-result["Distance to Default"])
    )


def test_merton_credit_spread_matches_financepy():
    # FinancePy MertonFirm(60, 50, 2, 0.03, 0.03, 0.35): equity_value, debt_value and
    # credit_spread. The equity and its volatility of that firm are the model's inputs.
    asset_value, asset_volatility, face, rate, years = 60.0, 0.35, 50.0, 0.03, 2.0
    d1 = (np.log(asset_value / face) + (rate + asset_volatility**2 / 2) * years) / (
        asset_volatility * np.sqrt(years)
    )
    equity = 18.119660661926517
    equity_volatility = norm.cdf(d1) * asset_volatility * asset_value / equity

    result = credit_model.get_merton_model(
        equity, equity_volatility, face, rate, time_horizon=years
    )

    assert result["Asset Value"] == pytest.approx(asset_value, rel=1e-6)
    assert result["Debt Value"] == pytest.approx(41.88033933807348, rel=1e-6)
    assert result["Credit Spread"] == pytest.approx(0.05860325841516467, rel=1e-5)
    assert result["Credit Spread"] == pytest.approx(
        -np.log(result["Debt Value"] / face) / years - rate
    )
    # The probability of default stays the last output, the score the factor analysis uses.
    assert result.index[-1] == "Probability of Default"


def test_merton_model_without_debt_or_data():
    assert (
        credit_model.get_merton_model(100, 0.4, 0, 0.04)["Probability of Default"] == 0
    )
    assert credit_model.get_merton_model(100, 0.4, 0, 0.04)["Credit Spread"] == 0
    assert np.isnan(
        credit_model.get_merton_model(np.nan, 0.4, 80, 0.04)["Distance to Default"]
    )


def test_synthetic_rating():
    table = pd.DataFrame(
        {
            "Lower": [-100000, 1, 3],
            "Upper": [0.999, 2.999, 100000],
            "Rating": ["B", "BBB", "AAA"],
            "Default Spread": [0.05, 0.01, 0.004],
        }
    )
    assert credit_model.get_synthetic_rating(0.5, table) == ("B", 0.05)
    assert credit_model.get_synthetic_rating(2.0, table) == ("BBB", 0.01)
    assert credit_model.get_synthetic_rating(float("inf"), table) == ("AAA", 0.004)
    assert credit_model.get_synthetic_rating(float("nan"), table)[0] == ""


def test_synthetic_rating_clips_to_the_table():
    table = pd.DataFrame(
        {
            "Lower": [-10, 1, 3],
            "Upper": [0.999, 2.999, 100000],
            "Rating": ["D", "BBB", "AAA"],
            "Default Spread": [0.19, 0.01, 0.004],
        }
    )
    # A coverage below the lowest bound (a large operating loss) is the worst rating.
    assert credit_model.get_synthetic_rating(-50.0, table) == ("D", 0.19)
    assert credit_model.get_synthetic_rating(1e9, table) == ("AAA", 0.004)
