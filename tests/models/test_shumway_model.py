"""Shumway Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.distress import shumway_model


def test_shumway_coefficients_are_those_of_table_6():
    # Shumway (2001), Table 6, panel B (market and accounting variables) and panel A
    # (market variables only).
    assert shumway_model.COEFFICIENTS["combined"] == {
        "Constant": -13.303,
        "NI/TA": -1.982,
        "TL/TA": 3.593,
        "RELSIZE": -0.467,
        "EXRET": -1.809,
        "SIGMA": 5.791,
    }
    assert shumway_model.COEFFICIENTS["market"]["Constant"] == -12.027
    assert shumway_model.COEFFICIENTS["market"]["RELSIZE"] == -0.503
    assert shumway_model.COEFFICIENTS["market"]["EXRET"] == -2.072
    assert shumway_model.COEFFICIENTS["market"]["SIGMA"] == 9.834


def test_get_shumway_score_and_probability():
    frame = pd.DataFrame
    score = shumway_model.get_shumway_score(
        net_income_to_total_assets=frame([[0.05]]),
        total_liabilities_to_total_assets=frame([[0.5]]),
        relative_size=frame([[-10.0]]),
        excess_return=frame([[0.0]]),
        sigma=frame([[0.1]]),
    )
    expected = -13.303 - 1.982 * 0.05 + 3.593 * 0.5 - 0.467 * -10 + 5.791 * 0.1
    assert score.iloc[0, 0] == pytest.approx(expected)
    probability = shumway_model.get_shumway_bankruptcy_probability(score)
    assert probability.iloc[0, 0] == pytest.approx(
        np.exp(expected) / (1 + np.exp(expected))
    )

    market = shumway_model.get_shumway_score(
        0.0, 0.0, frame([[-10.0]]), frame([[0.2]]), frame([[0.1]]), method="market"
    )
    assert market.iloc[0, 0] == pytest.approx(
        -12.027 - 0.503 * -10 - 2.072 * 0.2 + 9.834 * 0.1
    )


def test_excess_return_and_residual_sigma():
    rng = np.random.default_rng(11)
    market = rng.normal(0.01, 0.04, 12)
    company = 0.002 + 1.3 * market + rng.normal(0, 0.02, 12)

    def months(values):
        return [pd.DataFrame([[value]]) for value in values]

    excess = shumway_model.get_excess_return(months(company), months(market))
    assert excess.iloc[0, 0] == pytest.approx(
        np.prod(1 + company) - np.prod(1 + market)
    )

    sigma = shumway_model.get_residual_sigma(months(company), months(market))
    slope, intercept = np.polyfit(market, company, 1)
    residuals = company - intercept - slope * market
    assert sigma.iloc[0, 0] == pytest.approx(np.sqrt((residuals**2).sum() / 10))

    # A missing month leaves both NaN, as Shumway drops fewer than 12 months.
    company[3] = np.nan
    assert (
        shumway_model.get_residual_sigma(months(company), months(market))
        .isna()
        .all()
        .all()
    )
    assert (
        shumway_model.get_excess_return(months(company), months(market))
        .isna()
        .all()
        .all()
    )


def test_truncation_bounds_are_those_of_table_1():
    values = pd.DataFrame([[-1.0, 0.0, 1.0]])
    clipped = shumway_model.get_truncated_variable(values, "NI/TA")
    assert list(clipped.iloc[0]) == [-0.329, 0.0, 0.231]
