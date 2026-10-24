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


def test_gross_margin_and_altman_z_score_of_a_company_without_cost_of_goods(caplog):
    """A bank or insurer sells no goods, so its gross margin is left out."""
    toolkit = build_toolkit(rounding=4)
    toolkit._income_statement.loc[("MSFT", "Cost of Goods Sold"), :] = 0

    gross_margin = toolkit.ratios.get_gross_margin()

    assert gross_margin.loc["MSFT"].isna().all()
    assert gross_margin.loc["AAPL"].notna().any()

    toolkit.models.get_altman_z_score()

    assert "does not apply" in caplog.text
    assert "MSFT" in caplog.text


def test_an_annualized_sharpe_ratio(unrounded_toolkit, monkeypatch):
    import numpy as np

    sharpe = unrounded_toolkit.performance.get_sharpe_ratio()
    annualized = unrounded_toolkit.performance.get_sharpe_ratio(annualize=True)

    pd.testing.assert_frame_equal(annualized, sharpe * np.sqrt(252))

    # A probability is not annualized.
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")

    with pytest.raises(ValueError, match="annualized"):
        unrounded_toolkit.performance.get_sharpe_ratio(
            method="probabilistic", annualize=True
        )


def test_the_largest_holdings_of_a_fund(unrounded_toolkit):
    toolkit = build_toolkit(rounding=4)
    toolkit._api_key = "KEY"
    toolkit._etf_holdings = pd.DataFrame(
        {"Name": ["A", "B", "C", "D"], "Weight": [0.1, 0.4, 0.3, 0.2]},
        index=pd.MultiIndex.from_product([["FUND"], ["A", "B", "C", "D"]]),
    )
    toolkit._tickers = ["FUND"]

    assert list(toolkit.etf.get_holdings(limit=2).index) == ["B", "C"]
    assert list(toolkit.etf.get_holdings(limit=-2).index) == ["A", "D"]
    assert len(toolkit.etf.get_holdings()) == 4  # noqa: PLR2004


def test_the_key_maturities_of_the_eiopa_curve(monkeypatch):
    from financetoolkit.fixedincome import eiopa_model
    from financetoolkit.fixedincome.fixedincome_controller import FixedIncome

    curve = pd.DataFrame(
        [[0.02] * 150],
        index=pd.PeriodIndex(["2026-09"], freq="M"),
        columns=pd.MultiIndex.from_product(
            [["Netherlands"], [f"{year}Y" for year in range(1, 151)]],
            names=["Country", "Maturity"],
        ),
    )
    monkeypatch.setattr(
        eiopa_model, "get_risk_free_rate_term_structures", lambda *_, **__: curve
    )

    rates = FixedIncome(
        start_date="2026-01-01", end_date="2026-12-31", use_cached_data=False
    ).get_eiopa_risk_free_rate(maturity_years=[1, 10, 30])

    assert list(rates.columns.get_level_values(1)) == ["1Y", "10Y", "30Y"]
