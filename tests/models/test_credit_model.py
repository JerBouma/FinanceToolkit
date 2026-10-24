"""Credit Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from financetoolkit.models import credit_model

# pylint: disable=missing-function-docstring


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


def test_merton_model_without_debt_or_data():
    assert (
        credit_model.get_merton_model(100, 0.4, 0, 0.04)["Probability of Default"] == 0
    )
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
