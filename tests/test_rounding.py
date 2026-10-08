"""Rounding Tests

rounding=0 rounds to whole numbers, and rounding=None, on a method or on the Toolkit,
leaves the results unrounded, in every module.
"""

# pylint: disable=missing-function-docstring,protected-access

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Toolkit


def build_toolkit(rounding: int | None) -> Toolkit:
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
        rounding=rounding,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    return toolkit


CALLS = {
    "liquidity ratios": lambda toolkit, **kwargs: toolkit.ratios.collect_liquidity_ratios(
        **kwargs
    ),
    "valuation ratios": lambda toolkit, **kwargs: toolkit.ratios.collect_valuation_ratios(
        **kwargs
    ),
    "altman z-score": lambda toolkit, **kwargs: toolkit.models.get_altman_z_score(
        **kwargs
    ),
    "rsi": lambda toolkit, **kwargs: toolkit.technicals.get_relative_strength_index(
        **kwargs
    ),
    "black scholes": lambda toolkit, **kwargs: toolkit.options.get_black_scholes_model(
        **kwargs
    ),
    "delta": lambda toolkit, **kwargs: toolkit.options.get_delta(**kwargs),
}


def decimals(result: pd.Series | pd.DataFrame) -> int:
    """The largest number of decimals in the result."""
    values = np.asarray(result, dtype=float).ravel()
    values = values[np.isfinite(values)]

    return max(len(f"{value:.10f}".rstrip("0").split(".")[1]) for value in values)


@pytest.fixture(scope="module")
def unrounded_toolkit():
    return build_toolkit(rounding=None)


@pytest.mark.parametrize("name", CALLS)
def test_rounding_zero_rounds_to_whole_numbers(unrounded_toolkit, name):
    assert decimals(CALLS[name](unrounded_toolkit, rounding=0)) == 0


@pytest.mark.parametrize("name", CALLS)
def test_a_toolkit_without_rounding_does_not_round(unrounded_toolkit, name):
    assert decimals(CALLS[name](unrounded_toolkit)) > 4  # noqa: PLR2004
