"""Models Controller Tests""" ""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.utilities.error_model import InvalidArgumentError

# The GNP implicit price deflator of the U.S. Bureau of Economic Analysis (NIPA table 1.7.9,
# 2017 = 100) for the years the offline datasets cover, which the Ohlson O-Score deflates the
# total assets with.
GNP_DEFLATOR = pd.DataFrame(
    {
        "United States": [
            18.246,
            97.339,
            98.262,
            100.0,
            102.225,
            103.909,
            105.306,
            110.081,
            117.942,
            122.361,
            125.496,
            128.801,
        ]
    },
    index=pd.PeriodIndex(
        ["1968"] + [str(year) for year in range(2015, 2026)], freq="Y", name="Date"
    ),
)


@pytest.fixture(autouse=True)
def offline_gnp_deflator(monkeypatch):
    """Serves the GNP deflator of the Ohlson O-Score without a download."""
    from financetoolkit.economics.sources import bea_model  # noqa: PLC0415

    monkeypatch.setattr(
        bea_model, "get_gross_national_product_deflator", lambda: GNP_DEFLATOR
    )


def test_get_dupont_analysis(recorder, models_module):
    recorder.capture(models_module.get_dupont_analysis())
    recorder.capture(models_module.get_dupont_analysis(growth=True))
    recorder.capture(models_module.get_dupont_analysis(growth=True, lag=[1, 2, 3]))
    recorder.capture(models_module.get_dupont_analysis(method="extended"))
    recorder.capture(models_module.get_dupont_analysis(method="extended", growth=True))
    recorder.capture(
        models_module.get_dupont_analysis(method="extended", growth=True, lag=[1, 2, 3])
    )
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        models_module.get_dupont_analysis(method="nissim_penman")


def test_get_enterprise_value_breakdown(recorder, models_module):
    recorder.capture(models_module.get_enterprise_value_breakdown())
    recorder.capture(models_module.get_enterprise_value_breakdown(growth=True))
    recorder.capture(
        models_module.get_enterprise_value_breakdown(growth=True, lag=[1, 2, 3])
    )


def test_get_weighted_average_cost_of_capital(recorder, models_module):
    recorder.capture(models_module.get_weighted_average_cost_of_capital().round(1))
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True).round(1)
    )
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True, lag=[1, 2, 3])
    )


def test_get_intrinsic_valuation(recorder, models_module):
    recorder.capture(
        models_module.get_intrinsic_valuation(
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
        )
    )
    recorder.capture(
        models_module.get_intrinsic_valuation(
            growth_rate=[0.05, 0.08],
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
        )
    )


def test_get_gordon_growth_model(recorder, models_module):
    recorder.capture(
        models_module.get_gordon_growth_model(rate_of_return=0.1, growth_rate=0.09)
    )
    recorder.capture(
        models_module.get_gordon_growth_model(
            rate_of_return=0.2, growth_rate=0.1, project_periods=10
        )
    )


def test_get_altman_z_score(recorder, models_module):
    recorder.capture(models_module.get_altman_z_score())
    recorder.capture(models_module.get_altman_z_score(growth=True))
    recorder.capture(models_module.get_altman_z_score(growth=True, lag=[1, 2, 3]))


def test_get_piotroski_score(recorder, models_module):
    recorder.capture(models_module.get_piotroski_score())


def test_get_present_value_of_growth_opportunities(recorder, models_module):
    recorder.capture(models_module.get_present_value_of_growth_opportunities())
    recorder.capture(
        models_module.get_present_value_of_growth_opportunities(calculate_daily=True)
    )


def test_get_economic_value_added(recorder, models_module):
    recorder.capture(models_module.get_economic_value_added())
    recorder.capture(models_module.get_economic_value_added(growth=True))


def test_get_beneish_m_score(recorder, models_module):
    recorder.capture(models_module.get_beneish_m_score())
    recorder.capture(models_module.get_beneish_m_score(growth=True))


def test_get_ohlson_o_score(recorder, models_module):
    recorder.capture(models_module.get_ohlson_o_score())
    recorder.capture(models_module.get_ohlson_o_score(growth=True))
    recorder.capture(models_module.get_ohlson_o_score(growth=True, lag=[1, 2, 3]))


def test_get_zmijewski_score(recorder, models_module):
    recorder.capture(models_module.get_zmijewski_score())
    recorder.capture(models_module.get_zmijewski_score(growth=True))
    recorder.capture(models_module.get_zmijewski_score(growth=True, lag=[1, 2, 3]))


def test_get_sustainable_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_sustainable_growth_rate())
    recorder.capture(models_module.get_sustainable_growth_rate(growth=True))


def test_get_internal_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_internal_growth_rate())
    recorder.capture(models_module.get_internal_growth_rate(growth=True))


def test_get_market_value_added(recorder, models_module):
    recorder.capture(models_module.get_market_value_added())
    recorder.capture(models_module.get_market_value_added(growth=True))


def test_get_tobins_q_ratio(recorder, models_module):
    recorder.capture(models_module.get_tobins_q_ratio())
    recorder.capture(models_module.get_tobins_q_ratio(growth=True))


def test_get_free_cash_flow_to_firm(recorder, models_module):
    recorder.capture(models_module.get_free_cash_flow_to_firm())
    recorder.capture(models_module.get_free_cash_flow_to_firm(growth=True))


def test_get_free_cash_flow_to_equity(recorder, models_module):
    recorder.capture(models_module.get_free_cash_flow_to_equity())
    recorder.capture(models_module.get_free_cash_flow_to_equity(growth=True))


def test_get_two_stage_dividend_discount_model(recorder, models_module):
    recorder.capture(
        models_module.get_two_stage_dividend_discount_model(
            rate_of_return=0.10, high_growth_rate=0.12, stable_growth_rate=0.03
        )
    )
    recorder.capture(
        models_module.get_two_stage_dividend_discount_model(
            rate_of_return=[0.10, 0.09],
            high_growth_rate=0.12,
            stable_growth_rate=0.03,
            high_growth_periods=3,
        )
    )


def test_get_residual_income(recorder, models_module):
    recorder.capture(models_module.get_residual_income())
    recorder.capture(models_module.get_residual_income(growth=True))


def test_get_springate_score(recorder, models_module):
    recorder.capture(models_module.get_springate_score())
    recorder.capture(models_module.get_springate_score(growth=True))


def test_get_grover_score(recorder, models_module):
    recorder.capture(models_module.get_grover_score())
    recorder.capture(models_module.get_grover_score(growth=True))


def test_get_fulmer_h_score(recorder, models_module):
    recorder.capture(models_module.get_fulmer_h_score())
    recorder.capture(models_module.get_fulmer_h_score(growth=True))


def test_get_altman_z_score_models(recorder, models_module):
    for model in ("private", "non-manufacturing", "emerging"):
        recorder.capture(models_module.get_altman_z_score(method=model))

    # The four-variable models leave out sales, and the emerging market score is the
    # non-manufacturing score plus 3.25.
    non_manufacturing = models_module.get_altman_z_score(
        method="non-manufacturing", rounding=10
    )
    emerging = models_module.get_altman_z_score(method="emerging", rounding=10)
    assert "Sales to Total Assets" not in non_manufacturing.index.get_level_values(1)
    difference = (
        emerging.xs("Altman Z-Score", level=1)
        - non_manufacturing.xs("Altman Z-Score", level=1)
    ).stack()
    assert ((difference - 3.25).abs() < 1e-8).all()

    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        models_module.get_altman_z_score(method="zeta")


def test_get_taffler_z_score(recorder, models_module):
    recorder.capture(models_module.get_taffler_z_score())
    recorder.capture(models_module.get_taffler_z_score(growth=True))


def test_capital_asset_pricing_inputs_cover_the_trailing_span():
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    from financetoolkit.models.models_controller import Models
    from financetoolkit.models.valuation import intrinsic_model, wacc_model

    quarters = pd.period_range("2023Q1", "2024Q4", freq="Q")
    risk_free_rate = pd.Series(0.04, index=quarters)
    benchmark_returns = pd.Series(0.0225, index=quarters)

    # Quarterly data with a trailing 4 quarters: the flows are trailing twelve months, so
    # the rates are annual (4% and 4% + 5%), not the quarterly 1% and 2.25%.
    risk_free, market_return = Models._capital_asset_pricing_inputs(
        SimpleNamespace(_quarterly=True), risk_free_rate, benchmark_returns, 0.05, 4
    )
    assert risk_free.iloc[-1] == pytest.approx(0.04)
    assert market_return.iloc[-1] == pytest.approx(0.09)

    # Net Income of 100 over the year on a Book Value of 1000, with a Beta of 1.2, earns
    # exactly the annual Cost of Equity of 4% + 1.2 * 5% = 10%.
    beta = pd.DataFrame({"AAPL": 1.2}, index=quarters)
    cost_of_equity = wacc_model.get_cost_of_equity(risk_free, beta, market_return)
    assert cost_of_equity["AAPL"].iloc[-1] == pytest.approx(0.10)
    residual_income = intrinsic_model.get_residual_income(
        net_income=100,
        cost_of_equity=cost_of_equity["AAPL"].iloc[-1],
        book_value_of_equity=1000,
    )
    assert residual_income == pytest.approx(0)

    # Without trailing the rates stay per quarter, and the default premium (the average
    # benchmark return above the per-quarter rate, 1.25%, when no country premium is
    # known) scales the same way.
    no_country = SimpleNamespace(
        _quarterly=True, _country_risk_premiums=pd.Series({"AAPL": float("nan")})
    )
    risk_free, market_return = Models._capital_asset_pricing_inputs(
        no_country, risk_free_rate, benchmark_returns, None
    )
    assert risk_free.iloc[-1] == pytest.approx(0.01)
    assert market_return.iloc[-1] == pytest.approx(0.0225)
    risk_free, market_return = Models._capital_asset_pricing_inputs(
        no_country, risk_free_rate, benchmark_returns, None, 4
    )
    assert market_return.iloc[-1] == pytest.approx(0.09)


def test_cost_of_equity_follows_trailing(models_module):
    import pandas as pd

    # A trailing sum of two years of Net Income or Interest Expense is charged the Cost of
    # Equity of two years, in residual income and in the WACC (and so EVA and PVGO). Without
    # a premium the Cost of Equity is the risk-free rate, which doubles.
    single = models_module.get_residual_income(rounding=10, market_risk_premium=0).xs(
        "Cost of Equity", level=1
    )
    double = models_module.get_residual_income(
        rounding=10, trailing=2, market_risk_premium=0
    ).xs("Cost of Equity", level=1)
    pd.testing.assert_frame_equal(double, 2 * single, rtol=1e-6)

    wacc_single = models_module.get_weighted_average_cost_of_capital(
        rounding=10, market_risk_premium=0
    ).xs("Cost of Equity", level=1)
    wacc_double = models_module.get_weighted_average_cost_of_capital(
        rounding=10, trailing=2, market_risk_premium=0
    ).xs("Cost of Equity", level=1)
    pd.testing.assert_frame_equal(wacc_double, 2 * wacc_single, rtol=1e-6)

    # With a premium, both use the same Beta of the two years of daily returns.
    pd.testing.assert_frame_equal(
        models_module.get_residual_income(rounding=10, trailing=2).xs(
            "Cost of Equity", level=1
        ),
        models_module.get_weighted_average_cost_of_capital(rounding=10, trailing=2).xs(
            "Cost of Equity", level=1
        ),
        rtol=1e-6,
    )


def test_intrinsic_valuation_discounts_the_free_cash_flow_to_firm(models_module):
    import pandas as pd

    from financetoolkit.models.valuation import intrinsic_model

    valuation = models_module.get_intrinsic_valuation(
        growth_rate=0.05,
        perpetual_growth_rate=0.02,
        weighted_average_cost_of_capital=0.10,
        rounding=10,
    )
    fcff = models_module.get_free_cash_flow_to_firm(rounding=15).xs(
        "Free Cash Flow to Firm", level=1
    )

    for ticker in models_module._tickers:
        expected = intrinsic_model.get_intrinsic_value(
            cash_flow=fcff.loc[ticker].dropna().iloc[-1],
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
            cash_and_cash_equivalents=models_module._balance_sheet_statement.loc[
                ticker, "Cash and Cash Equivalents"
            ]
            .dropna()
            .iloc[-1],
            total_debt=models_module._balance_sheet_statement.loc[ticker, "Total Debt"]
            .dropna()
            .iloc[-1],
            shares_outstanding=models_module._income_statement.loc[
                ticker, "Weighted Average Shares Diluted"
            ]
            .dropna()
            .iloc[-1],
            # The marketable securities are added back with the cash, as the EV breakdown deducts both.
            short_term_investments=models_module._balance_sheet_statement.loc[
                ticker, "Short Term Investments"
            ]
            .dropna()
            .iloc[-1],
        ).round(10)
        pd.testing.assert_frame_equal(valuation.loc[ticker], expected)

    # The capital expenditure and working capital are not cash flows to value a company on.
    with pytest.raises(InvalidArgumentError, match="cash_flow_type must be one of"):
        models_module.get_intrinsic_valuation(
            0.05, 0.02, 0.10, cash_flow_type="Capital Expenditure"
        )
    # Without a discount rate above the perpetual growth rate there is no valuation.
    assert models_module.get_intrinsic_valuation(0.05, 0.10, 0.10).empty


def test_quarterly_valuations_use_the_trailing_twelve_months(
    models_module, monkeypatch
):
    import pandas as pd

    monkeypatch.setattr(models_module, "_quarterly", True)

    pd.testing.assert_frame_equal(
        models_module.get_intrinsic_valuation(0.05, 0.02, 0.10),
        models_module.get_intrinsic_valuation(0.05, 0.02, 0.10, trailing=4),
    )
    pd.testing.assert_frame_equal(
        models_module.get_graham_number(),
        models_module.get_graham_number(trailing=4),
    )


def test_present_value_of_growth_opportunities_uses_the_cost_of_equity(models_module):
    import pandas as pd

    pvgo = models_module.get_present_value_of_growth_opportunities(rounding=10)
    cost_of_equity = models_module.get_weighted_average_cost_of_capital(rounding=15).xs(
        "Cost of Equity", level=1
    )
    income = models_module._income_statement
    earnings_per_share = (
        income.loc[:, "Net Income", :]
        / income.loc[:, "Weighted Average Shares Diluted", :]
    )
    close = models_module._historical_data["yearly"]["Close"]

    expected = close[models_module._tickers] - (earnings_per_share / cost_of_equity).T
    expected = expected.loc[pvgo.index].round(10)
    pd.testing.assert_frame_equal(pvgo, expected, check_names=False, rtol=1e-8)


def test_residual_income_charges_the_beginning_book_value(models_module):
    import pandas as pd

    residual_income = models_module.get_residual_income(rounding=10)
    balance = models_module._balance_sheet_statement
    beginning_book_value = (
        balance.loc[:, "Total Shareholder Equity", :]
        - balance.loc[:, "Preferred Stock", :]
    ).shift(1, axis=1)

    book_value = residual_income.xs("Book Value of Equity", level=1)
    pd.testing.assert_frame_equal(
        book_value,
        beginning_book_value.loc[book_value.index, book_value.columns],
        check_names=False,
    )
    expected = (
        residual_income.xs("Net Income", level=1)
        - residual_income.xs("Cost of Equity", level=1) * book_value
    )
    pd.testing.assert_frame_equal(
        residual_income.xs("Residual Income", level=1), expected, rtol=1e-6
    )


def test_economic_value_added_uses_the_unrounded_wacc(models_module):
    import pandas as pd

    eva = models_module.get_economic_value_added(rounding=15)
    wacc = models_module.get_weighted_average_cost_of_capital(
        show_full_results=False, rounding=15
    )

    eva_wacc = eva.xs("Weighted Average Cost of Capital", level=1).dropna(
        axis=1, how="all"
    )
    pd.testing.assert_frame_equal(
        eva_wacc, wacc.loc[eva_wacc.index, eva_wacc.columns], check_names=False
    )


def test_models_take_operating_income_as_ebit(models_module):
    import pandas as pd

    altman = models_module.get_altman_z_score(rounding=10)
    income = models_module._income_statement
    balance = models_module._balance_sheet_statement
    expected = (
        income.loc[:, "Operating Income", :] / balance.loc[:, "Total Assets", :]
    ).round(10)

    ebit_to_assets = altman.xs("EBIT to Total Assets", level=1)
    pd.testing.assert_frame_equal(
        ebit_to_assets,
        expected.loc[ebit_to_assets.index, ebit_to_assets.columns],
        check_names=False,
    )


def test_growth_rates_use_beginning_equity_and_ending_assets(models_module):
    import pandas as pd

    income = models_module._income_statement
    balance = models_module._balance_sheet_statement
    cash = models_module._cash_flow_statement
    net_income = income.loc[:, "Net Income", :]
    retention = 1 - cash.loc[:, "Dividends Paid", :].abs() / net_income

    sustainable = models_module.get_sustainable_growth_rate(rounding=10)
    expected = (
        net_income / balance.loc[:, "Total Equity", :].shift(1, axis=1) * retention
    ).round(10)
    pd.testing.assert_frame_equal(
        sustainable,
        expected.loc[sustainable.index, sustainable.columns],
        check_names=False,
    )

    internal = models_module.get_internal_growth_rate(rounding=10)
    retained_return = net_income / balance.loc[:, "Total Assets", :] * retention
    expected = (retained_return / (1 - retained_return)).round(10)
    pd.testing.assert_frame_equal(
        internal, expected.loc[internal.index, internal.columns], check_names=False
    )


def test_dividend_discount_models_drop_an_unfinished_period():
    from types import SimpleNamespace

    import pandas as pd

    from financetoolkit.models.models_controller import Models

    years = pd.period_range("2022", "2024", freq="Y")
    dividends = pd.DataFrame({"AAPL": [0.9, 0.95, 0.5]}, index=years)

    def complete(last_day: str) -> pd.DataFrame:
        days = pd.period_range("2022-01-03", last_day, freq="D")
        historical = {
            "yearly": pd.concat({"Dividends": dividends}, axis=1),
            "daily": pd.DataFrame(index=days),
        }
        return Models._complete_period_dividends(
            SimpleNamespace(_quarterly=False, _historical_data=historical)
        )

    # Data up to the last trading day of the year covers the whole year.
    assert complete("2024-12-31").index[-1] == years[-1]
    # Data ending in June leaves 2024 unfinished, so 2023 is the last complete year.
    assert complete("2024-06-28").index[-1] == years[-2]


def test_operating_profit_after_taxes_uses_the_wacc_tax_rate(models_module):
    import pandas as pd

    from financetoolkit.models.valuation import wacc_model

    income = models_module._income_statement
    tax_rate = wacc_model.get_corporate_tax_rate(
        income.loc[:, "Income Tax Expense", :], income.loc[:, "Income Before Tax", :]
    )
    expected = (income.loc[:, "Operating Income", :] * (1 - tax_rate)).round(6)

    for result in (
        models_module.get_economic_value_added(rounding=6),
        models_module.get_free_cash_flow_to_firm(rounding=6),
    ):
        nopat = result.xs("Net Operating Profit After Taxes", level=1)
        pd.testing.assert_frame_equal(
            nopat,
            expected.loc[nopat.index, nopat.columns],
            check_names=False,
            check_freq=False,
        )


def test_quarterly_dividend_discount_models_use_an_annual_dividend(
    models_module, monkeypatch
):
    import pandas as pd
    import pytest

    from financetoolkit.models.valuation import intrinsic_model

    monkeypatch.setattr(models_module, "_quarterly", True)

    quarterly_dividends = models_module._complete_period_dividends()
    annual_dividend = quarterly_dividends.rolling(4).sum().iloc[-1]
    last_quarter = quarterly_dividends.index[-1]

    gordon = models_module.get_gordon_growth_model(
        rate_of_return=0.08, growth_rate=0.03, project_periods=2, rounding=10
    )
    # The last complete quarter is valued off the trailing four quarters of dividends.
    assert gordon.loc[last_quarter, "AAPL"] == pytest.approx(
        annual_dividend["AAPL"] * 1.03 / 0.05
    )
    # The projection steps are years, four quarters apart.
    assert list(gordon.index[-2:]) == [last_quarter + 4, last_quarter + 8]
    assert gordon.loc[last_quarter + 8, "AAPL"] == pytest.approx(
        annual_dividend["AAPL"] * 1.03**2 * 1.03 / 0.05
    )

    two_stage = models_module.get_two_stage_dividend_discount_model(
        0.08, 0.1, 0.03, rounding=10
    )
    expected = intrinsic_model.get_two_stage_dividend_discount_model(
        dividends_per_share=annual_dividend["MSFT"],
        rate_of_return=0.08,
        high_growth_rate=0.1,
        stable_growth_rate=0.03,
    ).round(10)
    pd.testing.assert_frame_equal(two_stage.loc["MSFT"], expected)


def test_graham_number_trails_only_the_earnings(models_module):
    import numpy as np
    import pandas as pd

    income = models_module._income_statement
    balance = models_module._balance_sheet_statement
    shares = income.loc[:, "Weighted Average Shares Diluted", :]

    earnings_per_share = (
        income.loc[:, "Net Income", :].T.rolling(2).sum() / shares.T.rolling(2).mean()
    ).T
    # Book value is point in time: the closing equity over the shares of that period.
    book_value_per_share = (
        balance.loc[:, "Total Shareholder Equity", :]
        - balance.loc[:, "Preferred Stock", :]
    ) / shares
    expected = np.sqrt(22.5 * earnings_per_share * book_value_per_share).round(6)

    graham = models_module.get_graham_number(trailing=2, rounding=6)
    pd.testing.assert_frame_equal(
        graham,
        expected.loc[graham.index, graham.columns],
        check_names=False,
        check_freq=False,
    )


def test_wacc_with_a_period_missing_from_the_balance_sheet():
    """A year the balance sheet lacks leaves the WACC of that year missing, not an error."""
    import pandas as pd  # noqa: PLC0415

    from financetoolkit import Toolkit  # noqa: PLC0415

    balance = pd.read_pickle("tests/datasets/balance_dataset.pickle")
    balance = balance.drop(columns=[c for c in balance.columns if str(c) == "2022"])
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        balance=balance,
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
        use_cached_data=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    wacc = toolkit.models.get_weighted_average_cost_of_capital()

    assert not wacc.empty
    assert wacc.loc[("AAPL", "Weighted Average Cost of Capital"), "2020"] > 0


def _use_quarterly_statements(models_module, monkeypatch, quarters=10):
    """Swap in ten quarters of statements built from the yearly ones, with a different
    scale per quarter so consecutive quarters and quarters a year apart differ."""
    import numpy as np
    import pandas as pd

    columns = pd.period_range("2020Q1", periods=quarters, freq="Q")
    factors = 1 + 0.1 * np.sin(np.arange(quarters) * 1.3) + 0.02 * np.arange(quarters)

    def to_quarterly(statement):
        values = np.column_stack(
            [
                statement.iloc[:, quarter % statement.shape[1]].to_numpy(dtype=float)
                * factor
                for quarter, factor in enumerate(factors)
            ]
        )
        # Scale the rows differently as well, so ratios of two items change over time.
        row_scale = 1 + 0.03 * np.cos(
            np.outer(np.arange(len(statement)), np.arange(quarters))
        )
        return pd.DataFrame(values * row_scale, index=statement.index, columns=columns)

    for name in [
        "_balance_sheet_statement",
        "_income_statement",
        "_cash_flow_statement",
    ]:
        monkeypatch.setattr(
            models_module, name, to_quarterly(getattr(models_module, name))
        )
    monkeypatch.setattr(models_module, "_quarterly", True)


def test_piotroski_score_compares_quarters_year_over_year(models_module, monkeypatch):
    yearly = models_module.get_piotroski_score()
    # On yearly data the change-based criteria compare with the previous column.
    balance = models_module._balance_sheet_statement
    current_ratio = (
        balance.loc[:, "Total Current Assets", :]
        / balance.loc[:, "Total Current Liabilities", :]
    )
    for ticker in models_module._tickers:
        for period in yearly.columns:
            assert yearly.loc[
                (ticker, "Change in Current Ratio Criteria"), period
            ] == int(
                current_ratio.loc[ticker, period]
                > current_ratio.loc[ticker, period - 1]
            )

    _use_quarterly_statements(models_module, monkeypatch)
    # trailing=1 scores single quarters (the default sums four), so the criteria can be
    # recomputed from the quarterly statements as they are.
    quarterly = models_module.get_piotroski_score(trailing=1)
    balance = models_module._balance_sheet_statement
    income = models_module._income_statement
    current_ratio = (
        balance.loc[:, "Total Current Assets", :]
        / balance.loc[:, "Total Current Liabilities", :]
    )
    gross_margin = (
        income.loc[:, "Revenue", :] - income.loc[:, "Cost of Goods Sold", :]
    ) / income.loc[:, "Revenue", :]

    # One quarter for the beginning total assets plus four for the year-over-year comparison.
    assert list(quarterly.columns) == list(balance.columns[5:])
    for ticker in models_module._tickers:
        for period in quarterly.columns:
            scores = quarterly.loc[ticker, period]
            assert scores["Change in Current Ratio Criteria"] == int(
                current_ratio.loc[ticker, period]
                > current_ratio.loc[ticker, period - 4]
            )
            assert scores["Gross Margin Criteria"] == int(
                gross_margin.loc[ticker, period] > gross_margin.loc[ticker, period - 4]
            )


def test_piotroski_score_trailing_uses_the_balances_a_year_apart(
    models_module, monkeypatch
):
    _use_quarterly_statements(models_module, monkeypatch)
    piotroski = models_module.get_piotroski_score(trailing=4)
    balance = models_module._balance_sheet_statement
    income = models_module._income_statement

    total_assets = balance.loc[:, "Total Assets", :]
    beginning_assets = total_assets.shift(4, axis=1)
    leverage = balance.loc[:, "Long Term Debt", :] / (
        (beginning_assets + total_assets) / 2
    )
    return_on_assets = (
        income.loc[:, "Net Income", :].T.rolling(4).sum().T / beginning_assets
    )

    assert not piotroski.empty
    for ticker in models_module._tickers:
        for period in piotroski.columns:
            scores = piotroski.loc[ticker, period]
            assert scores["Return on Assets Criteria"] == int(
                return_on_assets.loc[ticker, period] > 0
            )
            assert scores["Change in Leverage Criteria"] == int(
                leverage.loc[ticker, period] < leverage.loc[ticker, period - 4]
            )


def test_beneish_m_score_compares_quarters_year_over_year(models_module, monkeypatch):
    import pandas as pd

    yearly = models_module.get_beneish_m_score(rounding=10)
    revenue = models_module._income_statement.loc[:, "Revenue", :]
    pd.testing.assert_frame_equal(
        yearly.xs("Sales Growth Index", level=1).dropna(axis=1, how="all"),
        (revenue / revenue.shift(1, axis=1)).round(10).dropna(axis=1, how="all"),
        check_names=False,
    )

    _use_quarterly_statements(models_module, monkeypatch)
    quarterly = models_module.get_beneish_m_score(rounding=10, trailing=1)
    revenue = models_module._income_statement.loc[:, "Revenue", :]
    expected = (revenue / revenue.shift(4, axis=1)).round(10)

    pd.testing.assert_frame_equal(
        quarterly.xs("Sales Growth Index", level=1),
        expected.loc[:, quarterly.columns],
        check_names=False,
    )
    # The first year has nothing to compare with.
    m_score = quarterly.xs("Beneish M-Score", level=1)
    assert m_score.loc[:, : revenue.columns[3]].isna().all().all()
    assert m_score.loc[:, revenue.columns[4] :].notna().all().all()


def test_ohlson_o_score_compares_quarters_year_over_year(models_module, monkeypatch):
    import pandas as pd

    _use_quarterly_statements(models_module, monkeypatch)
    quarterly = models_module.get_ohlson_o_score(rounding=10, trailing=1)
    net_income = models_module._income_statement.loc[:, "Net Income", :]
    prior = net_income.shift(4, axis=1)
    expected = ((net_income - prior) / (net_income.abs() + prior.abs())).round(10)

    pd.testing.assert_frame_equal(
        quarterly.xs("Change in Net Income", level=1),
        expected.loc[:, quarterly.columns],
        check_names=False,
    )
    # Without a prior year the loss indicator is unknown, not 0.
    indicator = quarterly.xs("Negative Income Indicator", level=1)
    assert indicator.loc[:, : net_income.columns[3]].isna().all().all()


def test_get_reverse_discounted_cash_flow(recorder, models_module):
    recorder.capture(
        models_module.get_reverse_discounted_cash_flow(
            weighted_average_cost_of_capital=0.09
        )
    )
    recorder.capture(
        models_module.get_reverse_discounted_cash_flow(
            perpetual_growth_rate={"AAPL": 0.02, "MSFT": 0.03},
            weighted_average_cost_of_capital=0.1,
            cash_flow_type="Operating Cash Flow",
        )
    )


def test_get_earnings_power_value(recorder, models_module):
    recorder.capture(
        models_module.get_earnings_power_value(weighted_average_cost_of_capital=0.08)
    )
    recorder.capture(
        models_module.get_earnings_power_value(
            weighted_average_cost_of_capital=0.08, periods=2
        )
    )


def test_get_monte_carlo_intrinsic_valuation(recorder, models_module):
    recorder.capture(
        models_module.get_monte_carlo_intrinsic_valuation(
            growth_rate=0.08, weighted_average_cost_of_capital=0.09, simulations=2000
        )
    )


def test_get_merton_distance_to_default(recorder, models_module):
    recorder.capture(models_module.get_merton_distance_to_default())
    recorder.capture(models_module.get_merton_distance_to_default(time_horizon=2))


def test_get_synthetic_credit_rating(recorder, models_module, monkeypatch):
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models.valuation import damodaran_model  # noqa: PLC0415

    bounds = [
        -100000,
        0.2,
        0.65,
        0.8,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        3,
        4.25,
        5.5,
        6.5,
        8.5,
    ]
    ratings = [
        "D",
        "C",
        "CC",
        "CCC",
        "B-",
        "B",
        "B+",
        "BB",
        "BB+",
        "BBB",
        "A-",
        "A",
        "A+",
        "AA",
        "AAA",
    ]
    spreads = [
        0.19,
        0.16,
        0.13,
        0.09,
        0.05,
        0.03,
        0.028,
        0.018,
        0.014,
        0.011,
        0.009,
        0.008,
        0.007,
        0.0055,
        0.004,
    ]
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "Firm Type": firm_type,
                    "Lower": [bound * scale for bound in bounds],
                    "Upper": [bound * scale for bound in bounds[1:]] + [100000],
                    "Rating": ratings,
                    "Default Spread": spreads,
                }
            )
            for firm_type, scale in (("Large", 1), ("Small", 1.5), ("Financial", 0.3))
        ]
    )
    monkeypatch.setattr(damodaran_model, "get_synthetic_rating_table", lambda: table)

    recorder.capture(models_module.get_synthetic_credit_rating())
    recorder.capture(models_module.get_synthetic_credit_rating(firm_type="small"))
    with pytest.raises(InvalidArgumentError, match="firm_type must be one of"):
        models_module.get_synthetic_credit_rating(firm_type="tiny")


def test_get_peer_comparison_without_peer_data(models_module):
    assert models_module.get_peer_comparison().empty


def test_get_magic_formula(recorder, models_module):
    recorder.capture(models_module.get_magic_formula())


def test_get_mohanram_g_score(recorder, models_module):
    recorder.capture(models_module.get_mohanram_g_score(variance_periods=2))


def test_get_dechow_f_score(recorder, models_module):
    recorder.capture(models_module.get_dechow_f_score())


def test_get_montier_c_score(recorder, models_module):
    recorder.capture(models_module.get_montier_c_score())


def test_get_accruals_ratio(recorder, models_module):
    recorder.capture(models_module.get_accruals_ratio())


GROWTH = 1.03
QUARTERS = 20


def synthetic_models(items, quarterly=True):
    """
    A Models instance on synthetic statements, without the Toolkit around it. Items map a
    statement row to its values per ticker; the same frame serves as every statement.
    """
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models.models_controller import Models  # noqa: PLC0415

    tickers = sorted({ticker for values in items.values() for ticker in values})
    length = len(next(iter(next(iter(items.values())).values())))
    columns = pd.period_range(
        "2019Q1" if quarterly else "2010",
        periods=length,
        freq="Q" if quarterly else "Y",
    )
    statement = (
        pd.DataFrame(
            {
                (ticker, row): values[ticker]
                for row, values in items.items()
                for ticker in values
            },
            index=columns,
            dtype=float,
        )
        .T.sort_index()
        .rename_axis([None, None])
    )
    models = object.__new__(Models)
    models._tickers = tickers
    models._balance_sheet_statement = statement
    models._income_statement = statement
    models._cash_flow_statement = statement
    models._quarterly = quarterly
    models._rounding = 10
    models._start_date = None
    models._end_date = None
    models._period_end_dates_source = None
    models._historical_data = {
        "daily": pd.DataFrame(
            {("Close", ticker): [100.0] for ticker in tickers},
            index=pd.period_range("2023-12-29", periods=1, freq="D"),
        )
    }
    return models


def growing(start, length=QUARTERS):
    """Values that grow 3% a period."""
    return [start * GROWTH**period for period in range(length)]


def growing_company():
    """A company whose every item grows 3% a quarter, so 12.55% a year."""
    rows = {
        "Total Assets": 1000,
        "Total Current Assets": 400,
        "Total Current Liabilities": 200,
        "Total Liabilities": 500,
        "Cash and Short Term Investments": 100,
        "Cash and Cash Equivalents": 80,
        "Short Term Investments": 20,
        "Long Term Investments": 50,
        "Short Term Debt": 30,
        "Long Term Debt": 150,
        "Total Debt": 180,
        "Accounts Receivable": 60,
        "Inventory": 40,
        "Property, Plant and Equipment": 300,
        "Other Current Assets": 10,
        "Tax Payables": 5,
        "Revenue": 250,
        "Cost of Goods Sold": 150,
        "Operating Income": 50,
        "Income Before Tax": 40,
        "Income Tax Expense": 8,
        "Net Income": 25,
        "Research and Development Expenses": 12,
        "Selling and Marketing Expenses": 15,
        "Depreciation and Amortization": 10,
        "Operating Cash Flow": 30,
        "Free Cash Flow": 20,
        "Capital Expenditure": -10,
        "Cash Flow from Investing": -20,
    }
    items = {row: {"A": growing(start)} for row, start in rows.items()}
    items["Weighted Average Shares Diluted"] = {"A": [10.0] * QUARTERS}
    return items


def test_quarterly_scores_compare_year_over_year():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    models = synthetic_models(growing_company())
    g = GROWTH ** np.arange(QUARTERS)
    yearly_growth = GROWTH**4 - 1
    average_assets = 1000 * (g[-1] + g[-5]) / 2

    montier = models.get_montier_c_score(trailing=4).loc["A"]
    quarter_over_quarter = models.get_montier_c_score(trailing=1).loc["A"]
    # 12.55% a year is above Montier's 10%, 3% a quarter is not.
    assert montier.loc["Total Asset Growth"].iloc[-1] == 1
    assert quarter_over_quarter.loc["Total Asset Growth"].iloc[-1] == 0
    # A full trailing sum and the one a year back are needed for the first signs.
    assert str(montier.columns[0]) == "2020Q4"

    dechow = models.get_dechow_f_score(trailing=4).loc["A"]
    assert dechow.loc["Change in Cash Sales"].iloc[-1] == pytest.approx(yearly_growth)
    assert dechow.loc["Change in Receivables"].iloc[-1] == pytest.approx(
        60 * (g[-1] - g[-5]) / average_assets
    )
    assert dechow.loc["Soft Assets"].iloc[-1] == pytest.approx(
        (1000 - 300 - 100) / 1000
    )

    accruals = models.get_accruals_ratio(trailing=4).loc["A"]
    assert accruals.loc["Cash Flow Accruals Ratio"].iloc[-1] == pytest.approx(
        (25 - 30 + 20) * g[-4:].sum() / average_assets
    )
    balance_sheet = (
        (400 - 80) * (g[-1] - g[-5])
        - (200 - 30 - 5) * (g[-1] - g[-5])
        - 10 * g[-4:].sum()
    )
    assert accruals.loc["Balance Sheet Accruals Ratio"].iloc[-1] == pytest.approx(
        balance_sheet / average_assets
    )

    # Three yearly growth rates of trailing sums take sixteen quarters.
    mohanram = models.get_mohanram_g_score(trailing=4).loc["A"]
    assert str(mohanram.columns[0]) == "2022Q4"
    assert mohanram.loc["G-Score"].notna().all()


def test_quarterly_scores_default_to_a_trailing_year():
    import pandas as pd  # noqa: PLC0415

    models = synthetic_models(growing_company())
    for method in (
        "get_dechow_f_score",
        "get_montier_c_score",
        "get_mohanram_g_score",
        "get_accruals_ratio",
    ):
        pd.testing.assert_frame_equal(
            getattr(models, method)(), getattr(models, method)(trailing=4)
        )


def test_default_wacc_covers_a_year():
    import pandas as pd  # noqa: PLC0415

    models = synthetic_models(growing_company())
    requested = []

    def weighted_average_cost_of_capital(trailing=None, **kwargs):  # noqa: ARG001
        requested.append(trailing)
        return pd.DataFrame(
            {"2023Q4": [0.09]},
            index=pd.MultiIndex.from_tuples(
                [("A", "Weighted Average Cost of Capital")]
            ),
        )

    models._weighted_average_cost_of_capital = weighted_average_cost_of_capital

    models.get_earnings_power_value()
    models.get_reverse_discounted_cash_flow()
    models.get_monte_carlo_intrinsic_valuation(growth_rate=0.05, simulations=10)
    models.get_earnings_power_value(trailing=2)
    models.get_reverse_discounted_cash_flow(trailing=2)
    models.get_monte_carlo_intrinsic_valuation(
        growth_rate=0.05, simulations=10, trailing=8
    )

    # The WACC is a yearly rate: a year of interest expense on quarterly data, whatever
    # the trailing of the cash flows.
    assert requested == [4, 4, 4, 4, 4, 4]


def test_quality_scores_use_gross_debt_issuance_and_fixed_assets():
    items = {
        "Total Assets": {"A": [1000.0] * 3, "B": [1000.0] * 3},
        "Property, Plant and Equipment": {
            "A": [500.0, 500.0, 400.0],
            "B": [500.0, 500.0, 400.0],
        },
        "Gross Property, Plant and Equipment": {"A": [1000.0] * 3},
        "Depreciation and Amortization": {
            "A": [100.0, 100.0, 90.0],
            "B": [100.0, 100.0, 90.0],
        },
        "Gross Long Term Debt Issued": {"A": [0.0, 0.0, 50.0]},
        "Long Term Debt Issued": {"A": [0.0, 0.0, -10.0], "B": [0.0, 0.0, -10.0]},
        "Common Stock Issued": {"A": [0.0] * 3, "B": [0.0] * 3},
        "Revenue": {"A": [100.0] * 3, "B": [100.0] * 3},
    }
    models = synthetic_models(items, quarterly=False)

    # Both issued 50 of debt and repaid 60; only the gross issuance, which A reports,
    # shows the issuance.
    dechow = models.get_dechow_f_score()
    assert dechow.loc[("A", "Actual Issuance")].iloc[-1] == 1
    assert dechow.loc[("B", "Actual Issuance")].iloc[-1] == 0

    # Depreciation fell relative to gross fixed assets (10% to 9%) while it rose relative
    # to the net amount (20% to 22.5%), which only the gross amount reveals.
    montier = models.get_montier_c_score()
    assert montier.loc[("A", "Depreciation to Fixed Assets")].iloc[-1] == 1
    assert montier.loc[("B", "Depreciation to Fixed Assets")].iloc[-1] == 0


def test_quarterly_valuations_use_a_year_of_cash_flow_and_revenue():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    from financetoolkit.models.valuation import intrinsic_model  # noqa: PLC0415

    models = synthetic_models(growing_company())
    g = GROWTH ** np.arange(QUARTERS)
    last_year = g[-4:].sum()

    epv = models.get_earnings_power_value(weighted_average_cost_of_capital=0.1)
    assert epv.loc["A", "Normalized Operating Income"] == pytest.approx(
        0.2 * 250 * last_year
    )
    assert epv.loc["A", "Normalized NOPAT"] == pytest.approx(
        0.2 * 250 * last_year * 0.8
    )

    reverse = models.get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09, cash_flow_type="Free Cash Flow"
    )
    assert reverse.loc["A", "Cash Flow"] == pytest.approx(20 * last_year)
    # Trailing sums that grow 3% a quarter grow 12.55% a year.
    assert reverse.loc["A", "Historical Growth Rate"] == pytest.approx(GROWTH**4 - 1)

    simulated = models.get_monte_carlo_intrinsic_valuation(
        growth_rate=0.05,
        growth_rate_volatility=0,
        perpetual_growth_rate=0.02,
        perpetual_growth_rate_volatility=0,
        weighted_average_cost_of_capital=0.09,
        weighted_average_cost_of_capital_volatility=0,
        simulations=10,
        cash_flow_type="Free Cash Flow",
    )
    expected = float(
        intrinsic_model.get_intrinsic_value(
            20 * last_year,
            0.05,
            0.02,
            0.09,
            80 * g[-1],
            180 * g[-1],
            10,
            5,
            short_term_investments=20 * g[-1],
        )
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    assert simulated.loc["A", "Mean"] == pytest.approx(expected)


def test_market_values_use_the_traded_close(models_module):
    import pytest  # noqa: PLC0415

    prices = models_module._period_prices()
    close = models_module._historical_data["yearly"]["Close"]
    period = prices.columns[prices.loc["AAPL"].notna()][-1]
    assert prices.loc["AAPL", period] == pytest.approx(close.loc[period, "AAPL"])
    assert models_module._latest_price("AAPL") == pytest.approx(
        models_module._historical_data["daily"]["Close"]["AAPL"].dropna().iloc[-1]
    )


def test_synthetic_credit_rating_without_interest_expense(models_module, monkeypatch):
    import copy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models.valuation import damodaran_model  # noqa: PLC0415

    table = pd.DataFrame(
        {
            "Firm Type": "Large",
            "Lower": [-10.0, 1.0, 5.0],
            "Upper": [0.999, 4.999, 100000.0],
            "Rating": ["D", "BBB", "AAA"],
            "Default Spread": [0.19, 0.011, 0.004],
        }
    )
    monkeypatch.setattr(damodaran_model, "get_synthetic_rating_table", lambda: table)

    models = copy.copy(models_module)
    income = models_module._income_statement.copy()
    period = income.columns[-1]
    # No interest expense with a positive EBIT is an infinite coverage (Damodaran): the
    # highest rating, whether the interest expense is zero or not reported, and also next to
    # debt whose interest the statements do not report separately.
    income.loc[("AAPL", "Interest Expense"), period] = 0.0
    income.loc[("MSFT", "Interest Expense"), period] = np.nan
    assert (
        models_module._balance_sheet_statement.loc[("AAPL", "Total Debt"), period] > 0
    )
    models._income_statement = income

    rating = models.get_synthetic_credit_rating(firm_type="large")
    assert rating.loc[("AAPL", "Rating"), period] == "AAA"
    assert rating.loc[("MSFT", "Rating"), period] == "AAA"
    assert rating.loc[("AAPL", "Default Spread"), period] == pytest.approx(0.004)
    # The infinite ratio itself shows as NaN.
    assert pd.isna(rating.loc[("AAPL", "Interest Coverage Ratio"), period])

    # Without interest but with an operating loss there is nothing the coverage says about a
    # rating, and without an EBIT there is no ratio: no rating rather than the best one.
    income.loc[("AAPL", "Operating Income"), period] = -1e9
    income.loc[("MSFT", "Operating Income"), period] = np.nan
    rating = models.get_synthetic_credit_rating(firm_type="large")
    assert period not in rating.columns or (
        pd.isna(rating.loc[("AAPL", "Rating"), period])
        and pd.isna(rating.loc[("MSFT", "Rating"), period])
    )


def test_valuations_default_to_the_free_cash_flow_to_firm():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    from financetoolkit.models.valuation import intrinsic_model  # noqa: PLC0415

    items = growing_company()
    # A use of cash of 5 for working capital: an increase in net working capital.
    items["Change in Working Capital"] = {"A": growing(-5)}
    models = synthetic_models(items)
    g = GROWTH ** np.arange(QUARTERS)
    last_year = g[-4:].sum()
    # NOPAT (Operating Income 50 taxed at 8 / 40) + D&A 10 - capex 10 - ΔNWC 5, before
    # the interest that the reported Free Cash Flow of 20 is after.
    fcff = (50 * 0.8 + 10 - 10 - 5) * last_year

    reverse = models.get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09
    )
    assert reverse.loc["A", "Cash Flow"] == pytest.approx(fcff)

    simulated = models.get_monte_carlo_intrinsic_valuation(
        growth_rate=0.05,
        growth_rate_volatility=0,
        perpetual_growth_rate=0.02,
        perpetual_growth_rate_volatility=0,
        weighted_average_cost_of_capital=0.09,
        weighted_average_cost_of_capital_volatility=0,
        simulations=10,
    )
    expected = float(
        intrinsic_model.get_intrinsic_value(
            fcff,
            0.05,
            0.02,
            0.09,
            80 * g[-1],
            180 * g[-1],
            10,
            5,
            short_term_investments=20 * g[-1],
        )
        .loc["Intrinsic Value"]
        .iloc[0]
    )
    assert simulated.loc["A", "Mean"] == pytest.approx(expected)

    # Data errors give an empty result, which an unknown cash flow type is as well.
    assert models.get_reverse_discounted_cash_flow(cash_flow_type="Net Income").empty


def test_short_quarterly_scores_warn_how_many_quarters_they_need(monkeypatch):
    from financetoolkit.models import models_controller  # noqa: PLC0415

    items = {
        row: {ticker: values[:6] for ticker, values in tickers.items()}
        for row, tickers in growing_company().items()
    }
    models = synthetic_models(items)
    warnings = []
    monkeypatch.setattr(
        models_controller.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    for method, quarters in (
        ("get_dechow_f_score", 9),
        ("get_montier_c_score", 8),
        ("get_mohanram_g_score", 16),
    ):
        warnings.clear()
        assert getattr(models, method)().empty
        assert len(warnings) == 1
        assert f"at least {quarters} quarters" in warnings[0]
        assert "start_date" in warnings[0]

    # With enough quarters there is nothing to warn about.
    warnings.clear()
    synthetic_models(growing_company()).get_mohanram_g_score()
    assert warnings == []


def test_quarterly_valuations_skip_trailing_sums_with_a_missing_quarter():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    g = GROWTH ** np.arange(QUARTERS)

    # The latest quarter but two is missing: the trailing sums that include it are not
    # stretched over five quarters, so the latest complete one is that up to quarter 16.
    items = growing_company()
    items["Free Cash Flow"]["A"][17] = np.nan
    reverse = synthetic_models(items).get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09, cash_flow_type="Free Cash Flow"
    )
    assert reverse.loc["A", "Cash Flow"] == pytest.approx(20 * g[13:17].sum())

    # A quarter missing in between: the years run from the first to the latest trailing
    # sum on the calendar (four), not over the number of sums that remain.
    items = growing_company()
    items["Free Cash Flow"]["A"][8] = np.nan
    reverse = synthetic_models(items).get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09, cash_flow_type="Free Cash Flow"
    )
    assert reverse.loc["A", "Cash Flow"] == pytest.approx(20 * g[-4:].sum())
    assert reverse.loc["A", "Historical Growth Rate"] == pytest.approx(GROWTH**4 - 1)
    # The period names the first and latest trailing sum the growth is measured over.
    assert reverse.loc["A", "Historical Growth Period"] == "2019Q4-2023Q4"

    # The margins a year apart are taken at calendar positions counted back from the
    # latest quarter; the year that includes the missing revenue is left out instead of
    # shifting the steps onto overlapping windows.
    items = growing_company()
    margins = 0.1 + 0.01 * np.arange(QUARTERS)
    items["Operating Income"]["A"] = list(np.array(items["Revenue"]["A"]) * margins)
    items["Revenue"]["A"][9] = np.nan
    epv = synthetic_models(items).get_earnings_power_value(
        weighted_average_cost_of_capital=0.1
    )
    revenue = 250 * g
    operating_income = revenue * margins

    def trailing_margin(end):
        window = slice(end - 3, end + 1)
        return operating_income[window].sum() / revenue[window].sum()

    expected = np.mean([trailing_margin(end) for end in (3, 7, 15, 19)])
    assert epv.loc["A", "Average Operating Margin"] == pytest.approx(expected)


def test_magic_formula_on_a_short_quarterly_window(monkeypatch):
    import pandas as pd  # noqa: PLC0415

    from financetoolkit.models import models_controller  # noqa: PLC0415

    warnings = []
    monkeypatch.setattr(
        models_controller.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    def models_with(quarters):
        items = {
            row: {ticker: values[:quarters] for ticker, values in tickers.items()}
            for row, tickers in growing_company().items()
        }
        models = synthetic_models(items)
        models._historical_data["quarterly"] = pd.DataFrame(
            {("Close", "A"): [100.0] * quarters},
            index=models._balance_sheet_statement.columns,
        )
        return models

    # Three quarters hold no trailing twelve months: an empty frame and a warning.
    result = models_with(3).get_magic_formula()
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert len(warnings) == 1
    assert "at least 4 quarters" in warnings[0]

    # From the fourth quarter on, a frame with a rank per period and no warning.
    warnings.clear()
    result = models_with(6).get_magic_formula()
    assert list(result.columns.astype(str)) == ["2019Q4", "2020Q1", "2020Q2"]
    assert result.loc[("A", "Magic Formula Rank")].notna().all()
    assert warnings == []


def test_monte_carlo_valuation_without_a_share_price(models_module, monkeypatch):
    import pandas as pd  # noqa: PLC0415

    ticker = models_module._tickers[0]
    latest_price = models_module._latest_price
    monkeypatch.setattr(
        models_module,
        "_latest_price",
        lambda symbol: float("nan") if symbol == ticker else latest_price(symbol),
    )

    valuation = models_module.get_monte_carlo_intrinsic_valuation(
        growth_rate=0.08, weighted_average_cost_of_capital=0.09, simulations=200
    )

    assert pd.isna(valuation.loc[ticker, "Probability Undervalued"])
    assert valuation.drop(index=ticker)["Probability Undervalued"].notna().all()


def test_z_scores_describe_the_periods_shown(models_module, monkeypatch):
    from financetoolkit.utilities.statistics_model import (  # noqa: PLC0415
        calculate_standardization,
    )

    periods = models_module.get_dupont_analysis().columns
    monkeypatch.setattr(models_module, "_start_date", str(periods[2]))

    shown = models_module.get_dupont_analysis()
    standardized = models_module.get_dupont_analysis(standardize=True)

    assert standardized.columns[0] == periods[2]
    pd.testing.assert_frame_equal(
        standardized, calculate_standardization(shown, rounding=4, axis="columns")
    )


def test_cost_of_equity_uses_the_country_premium_of_each_company(test_toolkit):
    import numpy as np
    import pandas as pd

    def cost_of_equity(models, **kwargs):
        return models.get_weighted_average_cost_of_capital(rounding=None, **kwargs).xs(
            "Cost of Equity", level=1
        )

    historical = cost_of_equity(test_toolkit.models)

    models = test_toolkit.models
    models._country_risk_premiums_source = pd.Series({"AAPL": 0.05, "MSFT": np.nan})
    mixed = cost_of_equity(models)

    # AAPL takes its country's premium, MSFT (no premium known) the historical one.
    pd.testing.assert_series_equal(
        mixed.loc["AAPL"],
        cost_of_equity(test_toolkit.models, market_risk_premium=0.05).loc["AAPL"],
    )
    pd.testing.assert_series_equal(mixed.loc["MSFT"], historical.loc["MSFT"])


def test_toolkit_without_a_key_has_no_country_premiums(test_toolkit):
    assert test_toolkit._get_country_risk_premiums().isna().all()


def test_country_premiums_follow_the_premium_table_names(
    test_toolkit, monkeypatch, caplog
):
    import pandas as pd

    table = pd.DataFrame(
        {"Total Equity Risk Premium": [0.0433, 0.0487, 0.0584, 0.0499]},
        index=pd.Index(
            ["United States", "Korea", "Cayman Islands", "South Korea"], name="Country"
        ),
    )
    profile = pd.DataFrame({"AAPL": ["US"], "MSFT": ["KY"]}, index=["Country"])

    def get_profile():
        # The real profile marks and, with remove_invalid_tickers, drops invalid tickers.
        test_toolkit._invalid_tickers = ["MSFT"]
        test_toolkit._tickers = ["AAPL"]
        return profile

    tickers = list(test_toolkit._tickers)
    monkeypatch.setattr(test_toolkit, "_api_key", "key")
    monkeypatch.setattr(test_toolkit, "get_market_risk_premium", lambda: table)
    monkeypatch.setattr(test_toolkit, "get_profile", get_profile)

    premiums = test_toolkit._get_country_risk_premiums()

    assert premiums.to_dict() == {"AAPL": 0.0433, "MSFT": 0.0584}
    # The tickers stay; the invalid tickers the profile found stay recorded for get_profile.
    assert test_toolkit._tickers == tickers
    assert test_toolkit._invalid_tickers == ["MSFT"]

    # A country the table does not cover is named in a warning.
    profile.loc["Country", "MSFT"] = "VG"
    premiums = test_toolkit._get_country_risk_premiums()

    assert premiums.isna().to_dict() == {"AAPL": False, "MSFT": True}
    assert "VG" in caplog.text


def test_get_quality_score_needs_three_tickers(models_module, caplog):
    # The offline dataset holds two tickers, which cannot be ranked into z-scores.
    assert models_module.get_quality_score().empty
    assert "at least three tickers" in caplog.text


def test_get_h_model_values_every_period(models_module):
    h_model = models_module.get_h_model(
        rate_of_return=0.10, high_growth_rate=0.12, stable_growth_rate=0.03
    )
    dividends = models_module._annual_dividends_per_share().loc[h_model.index]

    expected = dividends * 1.03 / 0.07 + dividends * 5 * 0.09 / 0.07
    pd.testing.assert_frame_equal(
        h_model, expected.round(4)[h_model.columns], check_names=False
    )


def test_three_stage_model_adds_a_transition_phase(models_module):
    two_stage = models_module.get_two_stage_dividend_discount_model(
        rate_of_return=0.10, high_growth_rate=0.12, stable_growth_rate=0.03
    )
    three_stage = models_module.get_two_stage_dividend_discount_model(
        rate_of_return=0.10,
        high_growth_rate=0.12,
        stable_growth_rate=0.03,
        transition_periods=4,
    )

    # A slower decline to the stable rate keeps the growth higher for longer.
    assert (
        three_stage.xs("Intrinsic Value", level=1).iloc[:, 0]
        > two_stage.xs("Intrinsic Value", level=1).iloc[:, 0]
    ).all()


def test_get_justified_multiples(models_module):
    multiples = models_module.get_justified_multiples(
        rate_of_return=0.10, growth_rate=0.04
    )
    payout = multiples.xs("Payout Ratio", level=1)
    return_on_equity = multiples.xs("Return on Equity", level=1)

    pd.testing.assert_frame_equal(
        multiples.xs("Justified Leading P/E", level=1),
        (payout / 0.06).round(4),
        check_names=False,
        atol=1e-2,
    )
    pd.testing.assert_frame_equal(
        multiples.xs("Justified P/B", level=1),
        ((return_on_equity - 0.04) / 0.06).round(4),
        check_names=False,
        atol=1e-3,
    )


def test_historical_intrinsic_valuation_ends_at_the_single_valuation(models_module):
    single = models_module.get_intrinsic_valuation(
        growth_rate=0.05,
        perpetual_growth_rate=0.02,
        weighted_average_cost_of_capital=0.10,
    ).xs("Intrinsic Value", level=1)
    historical = models_module.get_intrinsic_valuation(
        growth_rate=0.05,
        perpetual_growth_rate=0.02,
        weighted_average_cost_of_capital=0.10,
        historical=True,
    )

    assert not historical.empty
    for ticker in historical.index:
        last_period = historical.loc[ticker].dropna().index[-1]
        if last_period == historical.columns[-1]:
            assert historical.loc[ticker, last_period] == single.loc[ticker].iloc[0]


def test_get_return_decomposition(recorder, models_module):
    recorder.capture(models_module.get_return_decomposition())
    recorder.capture(models_module.get_return_decomposition(logarithmic=True))


def test_default_growth_uses_no_later_periods(models_module):
    import pandas as pd
    import pytest

    growth = models_module._operating_growth()
    components = models_module.get_free_cash_flow_to_firm(rounding=15)
    reinvestment = (
        components.xs("Net Operating Profit After Taxes", level=1)
        - components.xs("Free Cash Flow to Firm", level=1)
    ).loc["AAPL"]
    invested_capital = (
        models_module._rows(
            models_module._balance_sheet_statement, "Total Shareholder Equity"
        )
        + models_module._rows(models_module._balance_sheet_statement, "Total Debt")
    ).loc["AAPL"]

    # Each period's growth only uses the reinvestment and capital up to that period.
    period = growth.columns[2]
    up_to = reinvestment.loc[:period].dropna()
    expected = (
        up_to.sum()
        / len(up_to)
        / invested_capital.reindex(reinvestment.index).loc[:period].mean()
    )
    assert growth.loc["AAPL", period] == pytest.approx(expected)

    # The stable growth stays within 0 and the long-run rate.
    stable = models_module._stable_growth(
        pd.DataFrame([[-0.1, 0.01, 0.2, float("nan")]], index=["AAPL"])
    ).loc["AAPL"]
    assert stable.tolist()[:4] == pytest.approx([0, 0.01, 0.025, 0.025])


def _models_without_benchmark():
    from financetoolkit import Toolkit  # noqa: PLC0415

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle").drop(
            columns="Benchmark", level=1
        ),
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        benchmark_ticker=None,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")
    return toolkit.models


def test_the_wacc_without_a_benchmark_is_reported_like_the_models_using_it(caplog):
    models = _models_without_benchmark()

    for metric in (
        models.get_weighted_average_cost_of_capital,
        models.get_economic_value_added,
        models.get_present_value_of_growth_opportunities,
    ):
        caplog.clear()
        assert metric().empty
        assert (
            f"benchmark_ticker in the Toolkit class to calculate {metric.__name__}"
            in (caplog.text)
        )


def test_the_wacc_keeps_no_results_between_calls(models_module):
    plain = models_module.get_weighted_average_cost_of_capital()

    models_module.get_weighted_average_cost_of_capital(growth=True, standardize=True)

    pd.testing.assert_frame_equal(
        models_module.get_weighted_average_cost_of_capital(), plain
    )
    assert not hasattr(models_module, "_weighted_average_cost_of_capital_growth")


def test_distress_scores_sum_a_year_of_quarters_by_default(models_module, monkeypatch):
    import pandas as pd

    _use_quarterly_statements(models_module, monkeypatch)

    # The scores were estimated on yearly statements, so quarterly data is a trailing year.
    for method in [
        models_module.get_altman_z_score,
        models_module.get_taffler_z_score,
        models_module.get_ohlson_o_score,
        models_module.get_springate_score,
    ]:
        pd.testing.assert_frame_equal(method(), method(trailing=4))


def test_get_campbell_hilscher_szilagyi_score(recorder, models_module, monkeypatch):
    # The S&P 500 offline: the benchmark of the pickled prices as both its price and its
    # total return index, and a total market value of 30 trillion on the last day.
    benchmark = models_module._historical_data["daily"]["Adj Close"]["Benchmark"]
    benchmark.index = benchmark.index.to_timestamp()

    def sp500(start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:  # noqa: ARG001
        index_data = pd.DataFrame(
            {"Price Index": benchmark, "Total Return Index": benchmark}
        )
        return index_data, pd.Series({benchmark.index[-1]: 3e13})

    monkeypatch.setattr(models_module, "_sp500_function", sp500)
    recorder.capture(models_module.get_campbell_hilscher_szilagyi_score())
    recorder.capture(models_module.get_campbell_hilscher_szilagyi_score(rounding=2))

    # Without the S&P 500, the relative size and the probability are NaN.
    monkeypatch.setattr(models_module, "_sp500_function", None)
    result = models_module.get_campbell_hilscher_szilagyi_score()
    assert result.xs("Failure Probability", level=1).isna().all().all()


def test_basket_models_leave_the_tickers_as_they_are(
    models_module, basket_toolkit, caplog
):
    basket_models = basket_toolkit.models

    # The Total is a company of its own in the scores built on its statements...
    piotroski = basket_models.get_piotroski_score()
    assert "Total" in piotroski.index.get_level_values(0)
    pd.testing.assert_frame_equal(
        piotroski.drop(index="Total", level=0), models_module.get_piotroski_score()
    )
    # ...but no peer of its own tickers in the scores that rank them against each other.
    magic_formula = basket_models.get_magic_formula()
    assert "Total" not in magic_formula.index.get_level_values(0)
    pd.testing.assert_frame_equal(magic_formula, models_module.get_magic_formula())
    assert basket_models.get_quality_score().empty
    assert "while 2 were given" in caplog.text

    # A value per share needs shares, which the Total has none of.
    intrinsic_valuation = basket_models.get_intrinsic_valuation()
    assert "Total" not in intrinsic_valuation.index.get_level_values(0)
    pd.testing.assert_frame_equal(
        intrinsic_valuation, models_module.get_intrinsic_valuation()
    )


def _fiscal_year_ends(models_module):
    """Apple's fiscal years end on 30 September and Microsoft's on 30 June."""
    columns = models_module._balance_sheet_statement.columns
    return pd.DataFrame(
        {
            period: [
                pd.Timestamp(f"{period.year}-09-30"),
                pd.Timestamp(f"{period.year}-06-30"),
            ]
            for period in columns
        },
        index=["AAPL", "MSFT"],
    )


def _daily(models_module, column, data=None):
    """A daily column of the historical data with a DatetimeIndex."""
    # Not `data or ...`: the risk free rate data is a LazyDict, empty (so falsy) until a
    # calculation first asks for a period.
    daily = (models_module._historical_data if data is None else data)["daily"][
        column
    ].copy()
    daily.index = daily.index.to_timestamp()
    return daily


def test_market_values_use_the_close_at_the_fiscal_period_end(
    models_module, monkeypatch
):
    dates = _fiscal_year_ends(models_module)
    monkeypatch.setattr(models_module, "_period_end_dates", dates)
    close = _daily(models_module, "Close")
    shares = models_module._income_statement.xs(
        "Weighted Average Shares Diluted", level=1
    )
    periods = [pd.Period("2021", freq="Y"), pd.Period("2022", freq="Y")]

    prices = models_module._period_prices()
    for ticker in ["AAPL", "MSFT"]:
        for period in periods:
            expected = close[ticker].loc[: dates.loc[ticker, period]].dropna().iloc[-1]
            assert prices.loc[ticker, period] == pytest.approx(expected)
    # The close of the last trading day of Apple's fiscal year 2021, not of December 2021.
    assert prices.loc["AAPL", periods[0]] == pytest.approx(141.5)
    assert prices.loc["MSFT", periods[0]] == pytest.approx(270.9)

    market_cap = prices * shares
    altman = models_module.get_altman_z_score(rounding=10)
    tobin = models_module.get_tobins_q_ratio(rounding=10)
    breakdown = models_module.get_enterprise_value_breakdown(rounding=10)
    wacc = models_module.get_weighted_average_cost_of_capital(rounding=10)
    mva = models_module.get_market_value_added(rounding=10)
    merton = models_module.get_merton_distance_to_default(rounding=10)
    multiples = models_module.get_justified_multiples(
        rate_of_return=0.1, growth_rate=0.02, rounding=10
    )
    total_liabilities = models_module._balance_sheet_statement.xs(
        "Total Liabilities", level=1
    )
    equity = models_module._balance_sheet_statement.xs(
        "Total Shareholder Equity", level=1
    )
    for ticker in ["AAPL", "MSFT"]:
        for period in periods:
            cap = market_cap.loc[ticker, period]
            assert altman.loc[
                (ticker, "Market Value to Total Liabilities"), period
            ] == pytest.approx(cap / total_liabilities.loc[ticker, period])
            assert tobin.loc[(ticker, "Market Value of Equity"), period] == (
                pytest.approx(cap)
            )
            assert breakdown.loc[(ticker, "Market Capitalization"), period] == (
                pytest.approx(cap)
            )
            assert wacc.loc[(ticker, "Market Value Equity"), period] == (
                pytest.approx(cap)
            )
            assert mva.loc[(ticker, "Market Value of Equity"), period] == (
                pytest.approx(cap)
            )
            assert merton.loc[(ticker, "Equity Value"), period] == pytest.approx(cap)
            assert multiples.loc[
                (ticker, "Price-to-Book Ratio"), period
            ] == pytest.approx(
                prices.loc[ticker, period]
                / (equity.loc[ticker, period] / shares.loc[ticker, period])
            )


def test_merton_measures_volatility_and_rates_at_the_fiscal_period_end(
    models_module, monkeypatch
):
    dates = _fiscal_year_ends(models_module)
    monkeypatch.setattr(models_module, "_period_end_dates", dates)
    returns = _daily(models_module, "Return")
    volatility = returns.rolling(252, min_periods=60).std() * 252**0.5
    risk_free = _daily(models_module, "Adj Close", models_module._risk_free_rate_data)

    merton = models_module.get_merton_distance_to_default(rounding=10)
    period = pd.Period("2021", freq="Y")
    for ticker in ["AAPL", "MSFT"]:
        date = dates.loc[ticker, period]
        assert merton.loc[(ticker, "Equity Volatility"), period] == pytest.approx(
            volatility[ticker].loc[:date].dropna().iloc[-1]
        )
    # The model is solved with the rate of the fiscal period end, converted from its
    # semi-annual quote to continuous compounding: the same inputs give the same distance to
    # default.
    from financetoolkit.models.distress import credit_model  # noqa: PLC0415
    from financetoolkit.options.options_model import (  # noqa: PLC0415
        get_continuously_compounded_rate,
    )

    date = dates.loc["AAPL", period]
    balance = models_module._balance_sheet_statement
    expected = credit_model.get_merton_model(
        equity_value=float(merton.loc[("AAPL", "Equity Value"), period]),
        equity_volatility=float(merton.loc[("AAPL", "Equity Volatility"), period]),
        default_point=float(
            balance.loc[("AAPL", "Short Term Debt"), period]
            + 0.5 * balance.loc[("AAPL", "Long Term Debt"), period]
        ),
        risk_free_rate=get_continuously_compounded_rate(
            float(risk_free.loc[:date].dropna().iloc[-1])
        ),
    )
    assert merton.loc[("AAPL", "Distance to Default"), period] == pytest.approx(
        expected["Distance to Default"]
    )


def test_present_value_of_growth_opportunities_uses_the_fiscal_period_end(
    models_module, monkeypatch
):
    period = pd.Period("2021", freq="Y")
    label = models_module.get_present_value_of_growth_opportunities(rounding=10)
    label_prices = models_module._period_prices()

    dates = _fiscal_year_ends(models_module)
    monkeypatch.setattr(models_module, "_period_end_dates", dates)
    fiscal = models_module.get_present_value_of_growth_opportunities(rounding=10)
    fiscal_prices = models_module._period_prices()

    # The no-growth value is the same; only the price is taken on another day.
    for ticker in ["AAPL", "MSFT"]:
        assert fiscal.loc[period, ticker] - label.loc[period, ticker] == (
            pytest.approx(
                fiscal_prices.loc[ticker, period] - label_prices.loc[ticker, period]
            )
        )

    # Daily, the fiscal year's earnings apply from the day after it ends, not from January.
    no_growth = fiscal_prices.loc["AAPL", period] - fiscal.loc[period, "AAPL"]
    daily = models_module.get_present_value_of_growth_opportunities(
        calculate_daily=True, rounding=10
    )
    close = models_module._historical_data["daily"]["Close"]["AAPL"]
    after = pd.Period("2021-10-01", freq="D")
    before = pd.Period("2021-09-29", freq="D")
    previous_no_growth = (
        fiscal_prices.loc["AAPL", period - 1] - fiscal.loc[period - 1, "AAPL"]
    )
    assert daily.loc[after, "AAPL"] == pytest.approx(close.loc[after] - no_growth)
    assert daily.loc[before, "AAPL"] == pytest.approx(
        close.loc[before] - previous_no_growth
    )


def test_return_decomposition_takes_the_dividends_of_the_fiscal_period(
    models_module, monkeypatch
):
    dates = _fiscal_year_ends(models_module)
    monkeypatch.setattr(models_module, "_period_end_dates", dates)
    dividends = _daily(models_module, "Dividends")
    prices = models_module._period_prices()

    decomposition = models_module.get_return_decomposition(rounding=10)
    period = pd.Period("2022", freq="Y")
    for ticker in ["AAPL", "MSFT"]:
        begin, end = dates.loc[ticker, period - 1], dates.loc[ticker, period]
        paid = dividends[ticker].loc[begin:end].drop(begin, errors="ignore").sum()
        assert decomposition.loc[(ticker, "Dividend Return"), period] == (
            pytest.approx(paid / prices.loc[ticker, period - 1])
        )
        assert decomposition.loc[(ticker, "Price Return"), period] == (
            pytest.approx(
                prices.loc[ticker, period] / prices.loc[ticker, period - 1] - 1
            )
        )


def test_return_on_equity_excludes_minority_interests(models_module, monkeypatch):
    import copy  # noqa: PLC0415

    models = copy.copy(models_module)
    balance = models_module._balance_sheet_statement.copy()
    # A minority interest makes the Total Equity larger than the shareholders' equity.
    for ticker in ["AAPL", "MSFT"]:
        balance.loc[(ticker, "Total Equity")] = (
            balance.loc[(ticker, "Total Shareholder Equity")] + 1e10
        )
    models._balance_sheet_statement = balance

    equity = balance.xs("Total Shareholder Equity", level=1)
    assets = balance.xs("Total Assets", level=1)
    net_income = models_module._income_statement.xs("Net Income", level=1)
    period = pd.Period("2022", freq="Y")
    average_equity = (equity[period] + equity[period - 1]) / 2
    average_assets = (assets[period] + assets[period - 1]) / 2

    dupont = models.get_dupont_analysis(rounding=10)
    extended = models.get_dupont_analysis(method="extended", rounding=10)
    for ticker in ["AAPL", "MSFT"]:
        for analysis in (dupont, extended):
            assert analysis.loc[(ticker, "Equity Multiplier"), period] == (
                pytest.approx(average_assets[ticker] / average_equity[ticker])
            )
            assert analysis.loc[(ticker, "Return on Equity"), period] == (
                pytest.approx(net_income.loc[ticker, period] / average_equity[ticker])
            )

    return_on_equity, _ = models._return_on_equity_and_payout(None)
    assert return_on_equity.loc["AAPL", period] == pytest.approx(
        net_income.loc["AAPL", period] / equity.loc["AAPL", period - 1]
    )


def test_trailing_scores_use_the_balances_at_the_period_end(models_module):
    balance = models_module._balance_sheet_statement
    period = pd.Period("2022", freq="Y")

    def closing(item):
        return balance.loc[("AAPL", item), period]

    # With trailing=2 the flows cover two years, while the balances, the shares and the
    # market values stay those at the end of the period.
    altman = models_module.get_altman_z_score(trailing=2, rounding=10)
    assert altman.loc[("AAPL", "Working Capital to Total Assets"), period] == (
        pytest.approx(
            (closing("Total Current Assets") - closing("Total Current Liabilities"))
            / closing("Total Assets")
        )
    )
    market_cap = (
        models_module._period_prices()
        * models_module._income_statement.xs("Weighted Average Shares Diluted", level=1)
    ).loc["AAPL", period]
    assert altman.loc[("AAPL", "Market Value to Total Liabilities"), period] == (
        pytest.approx(market_cap / closing("Total Liabilities"))
    )

    for method, item, row in [
        (models_module.get_ohlson_o_score, "Total Liabilities to Total Assets", None),
        (models_module.get_zmijewski_score, None, None),
        (models_module.get_springate_score, None, None),
        (models_module.get_grover_score, None, None),
        (models_module.get_fulmer_h_score, None, None),
        (models_module.get_taffler_z_score, None, None),
    ]:
        trailing = method(trailing=2, rounding=10)
        single = method(rounding=10)
        if item:
            assert trailing.loc[("AAPL", item), period] == pytest.approx(
                closing("Total Liabilities") / closing("Total Assets")
            )
        # Every component without a flow is the same as without trailing.
        balance_only = [
            component
            for component in single.loc["AAPL"].index
            if component
            in (
                "Total Liabilities to Total Assets",
                "Working Capital to Total Assets",
                "Current Liabilities to Current Assets",
                "Current Assets to Total Liabilities",
                "Current Liabilities to Total Assets",
                "Current Ratio",
                "Debt to Assets",
            )
        ]
        for component in balance_only:
            assert trailing.loc[("AAPL", component), period] == pytest.approx(
                single.loc[("AAPL", component), period]
            )

    tobin = models_module.get_tobins_q_ratio(rounding=10)
    assert tobin.loc[("AAPL", "Total Assets"), period] == closing("Total Assets")
    assert tobin.loc[("AAPL", "Market Value of Equity"), period] == (
        pytest.approx(market_cap)
    )

    wacc = models_module.get_weighted_average_cost_of_capital(
        trailing=2, rounding=10, market_risk_premium=0.05
    )
    assert wacc.loc[("AAPL", "Market Value Debt"), period] == closing("Total Debt")
    assert wacc.loc[("AAPL", "Market Value Equity"), period] == pytest.approx(
        market_cap
    )


def test_free_cash_flow_to_firm_clips_the_tax_rate():
    import numpy as np  # noqa: PLC0415

    items = growing_company()
    items["Change in Working Capital"] = {"A": growing(-5)}
    # A tax benefit on a profit: the corporate tax rate is 0, not negative, so the NOPAT
    # does not exceed the operating income.
    items["Income Tax Expense"] = {"A": growing(-8)}
    models = synthetic_models(items)
    last_year = (GROWTH ** np.arange(QUARTERS))[-4:].sum()

    reverse = models.get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09
    )
    assert reverse.loc["A", "Cash Flow"] == pytest.approx(
        (50 + 10 - 10 - 5) * last_year
    )


def test_ohlson_size_is_deflated_by_the_gnp_price_level(models_module, monkeypatch):
    import numpy as np  # noqa: PLC0415

    from financetoolkit.models.distress import distress_controller  # noqa: PLC0415

    total_assets = models_module._balance_sheet_statement.xs("Total Assets", level=1)
    ohlson = models_module.get_ohlson_o_score(rounding=10)
    deflator = GNP_DEFLATOR["United States"]
    for year in ["2021", "2022"]:
        period = pd.Period(year, freq="Y")
        index = deflator[period] / deflator[pd.Period("1968", freq="Y")] * 100
        assert ohlson.loc[("AAPL", "Log of Total Assets"), period] == pytest.approx(
            np.log(total_assets.loc["AAPL", period] / index)
        )

    # A year the deflator is not published for yet uses the latest published year.
    columns = pd.PeriodIndex(["2025Q4", "2026Q1"], freq="Q")
    latest = models_module._gross_national_product_price_level(columns)
    assert latest.iloc[1] == latest.iloc[0]

    # Without the deflator, the size and the score are NaN rather than on another scale.
    warnings = []
    monkeypatch.setattr(
        distress_controller.bea_model,
        "get_gross_national_product_deflator",
        pd.DataFrame,
    )
    monkeypatch.setattr(
        distress_controller.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    missing = models_module.get_ohlson_o_score()
    assert missing.xs("Log of Total Assets", level=1).isna().all().all()
    assert missing.xs("Ohlson O-Score", level=1).isna().all().all()
    assert missing.xs("Total Liabilities to Total Assets", level=1).notna().any().any()
    assert any("GNP price deflator" in warning for warning in warnings)


def test_market_value_added_compares_values_at_the_same_date(models_module):
    balance = models_module._balance_sheet_statement
    mva = models_module.get_market_value_added(rounding=10)
    period = pd.Period("2022", freq="Y")

    for ticker in ["AAPL", "MSFT"]:
        invested_capital = (
            balance.loc[(ticker, "Total Equity"), period]
            + balance.loc[(ticker, "Total Debt"), period]
        )
        assert mva.loc[(ticker, "Invested Capital"), period] == pytest.approx(
            invested_capital
        )
        # The debt at its book value cancels out: market value of equity minus book equity.
        assert mva.loc[(ticker, "Market Value Added"), period] == pytest.approx(
            mva.loc[(ticker, "Market Value of Equity"), period]
            - balance.loc[(ticker, "Total Equity"), period]
        )

    averaged = models_module.get_market_value_added(trailing=2, rounding=10)
    assert averaged.loc[("AAPL", "Invested Capital"), period] == pytest.approx(
        (
            balance.loc[("AAPL", "Total Equity"), [period - 1, period]].sum()
            + balance.loc[("AAPL", "Total Debt"), [period - 1, period]].sum()
        )
        / 2
    )


def test_the_wacc_with_debt_but_no_interest_expense_is_unknown(models_module):
    import copy  # noqa: PLC0415

    models = copy.copy(models_module)
    income = models_module._income_statement.copy()
    period = income.columns[-2]
    # A zero interest expense next to debt is interest the statements do not report.
    income.loc[("AAPL", "Interest Expense"), period] = 0.0
    models._income_statement = income

    wacc = models.get_weighted_average_cost_of_capital()

    assert pd.isna(wacc.loc[("AAPL", "Cost of Debt"), period])
    assert pd.isna(wacc.loc[("AAPL", "Weighted Average Cost of Capital"), period])
    assert wacc.loc[("MSFT", "Weighted Average Cost of Capital"), period] > 0


def _models_with_estimates(test_toolkit):
    """A Models instance with yearly consensus EPS estimates for 2017-2026."""
    models = test_toolkit.models
    years = pd.period_range("2017", periods=10, freq="Y")
    models._analyst_estimates = pd.DataFrame(
        [
            [1 + 0.1 * year for year in range(10)],
            [12] * 10,
            [2 + 0.3 * year for year in range(10)],
            [12] * 5 + [3] + [12] * 4,
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("AAPL", "Estimated EPS Average"),
                ("AAPL", "Number of Analysts EPS"),
                ("MSFT", "Estimated EPS Average"),
                ("MSFT", "Number of Analysts EPS"),
            ]
        ),
        columns=years,
    )
    return models


def test_get_implied_cost_of_capital(recorder, test_toolkit):
    models = _models_with_estimates(test_toolkit)

    recorder.capture(models.get_implied_cost_of_capital())
    recorder.capture(models.get_implied_cost_of_capital(modified=False))
    recorder.capture(models.get_implied_cost_of_capital(min_analysts=5))


def test_implied_cost_of_capital_uses_the_next_two_years_and_the_period_end_close(
    test_toolkit,
):
    models = _models_with_estimates(test_toolkit)
    result = models.get_implied_cost_of_capital(rounding=None)
    peg = models.get_implied_cost_of_capital(modified=False, rounding=None)

    # 2020: EPS1 the 2021 estimate (1.4), EPS2 the 2022 estimate (1.5), the close on
    # 2020-12-31 and the dividends paid over 2020.
    daily = test_toolkit._get_module_historical_data("daily")
    price = daily["Close"]["AAPL"].loc["2020-12-31"]
    dividends = daily["Dividends"]["AAPL"].loc["2020-01-01":"2020-12-31"].sum()
    half_yield = dividends / (2 * price)
    expected = half_yield + np.sqrt(half_yield**2 + 0.1 / price)

    assert result.loc["AAPL", "2020"] == pytest.approx(expected, abs=1e-4)
    assert peg.loc["AAPL", "2020"] == pytest.approx(np.sqrt(0.1 / price), abs=1e-4)
    # 2023 has no price in the offline data; MSFT 2021 needs the 2022 estimate, which
    # three analysts cover.
    assert result.loc[:, "2023"].isna().all()
    restricted = models.get_implied_cost_of_capital(min_analysts=5)
    assert pd.isna(restricted.loc["MSFT", "2021"])
    assert pd.isna(restricted.loc["MSFT", "2020"])
    assert not pd.isna(restricted.loc["MSFT", "2022"])


def test_implied_cost_of_capital_without_estimates_is_empty(test_toolkit):
    models = test_toolkit.models
    models._analyst_estimates = pd.DataFrame()

    assert models.get_implied_cost_of_capital().empty


def test_get_nissim_penman_decomposition(recorder, models_module):
    recorder.capture(models_module.get_nissim_penman_decomposition())
    recorder.capture(models_module.get_nissim_penman_decomposition(growth=True))


def test_nissim_penman_decomposition_equals_the_ratios(models_module, test_toolkit):
    decomposition = models_module.get_nissim_penman_decomposition(rounding=10)
    ratios = test_toolkit.ratios
    equity = models_module._balance_sheet_statement.xs(
        "Total Shareholder Equity", level=1
    ) - models_module._balance_sheet_statement.xs("Preferred Stock", level=1).fillna(0)
    net_income = models_module._income_statement.xs("Net Income", level=1)
    return_on_common_equity = net_income / equity.T.rolling(2).mean().T

    for name, method in [
        ("Return on Net Operating Assets", ratios.get_return_on_net_operating_assets),
        ("Net Borrowing Cost", ratios.get_net_borrowing_cost),
        ("Financial Leverage", ratios.get_financial_leverage),
    ]:
        expected = method(rounding=10)
        shown = decomposition.xs(name, level=1)
        pd.testing.assert_frame_equal(
            shown, expected.reindex_like(shown), check_names=False, atol=1e-9
        )

    shown = decomposition.xs("Return on Common Equity", level=1)
    pd.testing.assert_frame_equal(
        shown,
        return_on_common_equity.reindex_like(shown),
        check_names=False,
        atol=1e-9,
    )


def test_nissim_penman_decomposition_is_nan_without_interest_rows(models_module):
    import copy  # noqa: PLC0415

    models = copy.copy(models_module)
    income = models_module._income_statement.copy()
    period = pd.Period("2022", freq="Y")
    income.loc[("AAPL", "Interest Expense"), period] = 0
    income.loc[("AAPL", "Interest Income"), period] = 0
    models._income_statement = income

    base = models_module.get_nissim_penman_decomposition(rounding=10)
    decomposition = models.get_nissim_penman_decomposition(rounding=10)

    unknown = [
        "Return on Net Operating Assets",
        "Net Borrowing Cost",
        "Spread",
        "Leverage Effect",
        "Return on Common Equity",
    ]
    assert decomposition.loc[[("AAPL", name) for name in unknown], period].isna().all()
    assert decomposition.loc[("AAPL", "Financial Leverage"), period] == pytest.approx(
        base.loc[("AAPL", "Financial Leverage"), period]
    )
    # Every other cell is unchanged.
    pd.testing.assert_frame_equal(
        decomposition.drop(columns=period), base.drop(columns=period)
    )
    pd.testing.assert_series_equal(
        decomposition.loc["MSFT", period], base.loc["MSFT", period]
    )


# The US CPI-U of the BIS (2010 = 100) that the size-age index deflates the total assets with,
# rising 0.25% a month from its 2004 level.
CONSUMER_PRICES = pd.DataFrame(
    {"United States": [85.0 * 1.0025**month for month in range(240)]},
    index=pd.period_range("2004-01", periods=240, freq="M"),
)

PROFILE = pd.DataFrame(
    {
        "AAPL": {"IPO Date": "1980-12-12", "Industry": "Consumer Electronics"},
        "MSFT": {"IPO Date": "1986-03-13", "Industry": "Software - Infrastructure"},
    }
)


@pytest.fixture
def constraints_module(test_toolkit, monkeypatch):
    """A Models instance with company profiles and the CPI, without a download."""
    from financetoolkit.economics.sources import bis_model  # noqa: PLC0415

    monkeypatch.setattr(
        bis_model, "get_consumer_prices", lambda *args, **kwargs: CONSUMER_PRICES
    )
    module = test_toolkit.models
    module._profile_source = PROFILE
    return module


def test_get_kaplan_zingales_index(recorder, constraints_module):
    recorder.capture(constraints_module.get_kaplan_zingales_index())
    recorder.capture(constraints_module.get_kaplan_zingales_index(growth=True))
    recorder.capture(constraints_module.get_kaplan_zingales_index(standardize=True))


def test_get_whited_wu_index(recorder, constraints_module):
    recorder.capture(constraints_module.get_whited_wu_index())
    recorder.capture(constraints_module.get_whited_wu_index(growth=True))


def test_get_whited_wu_index_averages_the_industry(constraints_module):
    constraints_module._profile_source = PROFILE.replace(
        "Software - Infrastructure", "Consumer Electronics"
    )
    result = constraints_module.get_whited_wu_index()
    sales_growth = result.xs("Sales Growth", level=1)
    industry = result.xs("Industry Sales Growth", level=1)

    # Rounded to the Toolkit's 4 decimals after averaging.
    for ticker in ["AAPL", "MSFT"]:
        pd.testing.assert_series_equal(
            industry.loc[ticker], sales_growth.mean(), check_names=False, atol=1e-4
        )


def test_get_size_age_index(recorder, constraints_module):
    recorder.capture(constraints_module.get_size_age_index())
    recorder.capture(constraints_module.get_size_age_index(growth=True))


def test_get_size_age_index_without_a_profile_is_nan(constraints_module):
    constraints_module._profile_source = None
    result = constraints_module.get_size_age_index()

    assert result.empty or result.xs("Size-Age Index", level=1).isna().all().all()


def test_get_size_age_index_deflates_periods_before_the_base_year(
    constraints_module, monkeypatch
):
    import copy  # noqa: PLC0415
    import math  # noqa: PLC0415

    from financetoolkit.economics.sources import bis_model  # noqa: PLC0415

    # The CPI from 1990, 0.25% a month, returned from the requested start onwards.
    consumer_prices = pd.DataFrame(
        {"United States": [60.0 * 1.0025**month for month in range(480)]},
        index=pd.period_range("1990-01", periods=480, freq="M"),
    )
    requested = []

    def get_consumer_prices(measure, start_date, end_date):
        requested.append(start_date)
        return consumer_prices[consumer_prices.index >= pd.Period(start_date, "M")]

    monkeypatch.setattr(bis_model, "get_consumer_prices", get_consumer_prices)

    models = copy.copy(constraints_module)
    balance = constraints_module._balance_sheet_statement.copy()
    # The same statements labelled 1998 onwards, ending on the calendar year end.
    balance.columns = pd.period_range("1998", periods=len(balance.columns), freq="Y")
    # $100 million of assets, below the cap, so the Size moves with the price level.
    balance.loc[(slice(None), "Total Assets"), :] = 100e6
    models._balance_sheet_statement = balance
    models._period_end_dates = None

    size = models.get_size_age_index(rounding=10).xs("Size", level=1)

    assert requested[0] < "1998-01-01"
    base = consumer_prices.loc["2004"].mean().iloc[0]
    for period in size.columns:
        price_level = consumer_prices.loc[pd.Period(f"{period.year}-12", "M")].iloc[0]
        assert size.loc["AAPL", period] == pytest.approx(
            math.log(100 / (price_level / base))
        )


def test_get_size_age_index_does_not_warn_about_the_basket_total(
    constraints_module, caplog
):
    import copy  # noqa: PLC0415

    models = copy.copy(constraints_module)
    # The basket's Total has no profile, so no IPO date, and is no company.
    models._tickers = ["AAPL", "MSFT", "Total"]

    with caplog.at_level("WARNING"):
        models.get_size_age_index()

    assert "No IPO date" not in caplog.text

    models._profile = PROFILE.drop(columns="MSFT")
    with caplog.at_level("WARNING"):
        models.get_size_age_index()

    assert "No IPO date is available for MSFT," in caplog.text


def test_get_graham_intrinsic_value(recorder, models_module, monkeypatch):
    import pandas as pd

    # Moody's Aaa yield offline: 4% rising by a basis point a month.
    days = pd.date_range("2015-01-01", "2023-12-31", freq="B")
    aaa = pd.Series(
        0.04 + 0.0001 * ((days.year - 2015) * 12 + days.month - 1), index=days
    )
    monkeypatch.setattr(
        models_module,
        "_corporate_bond_yield_function",
        lambda start, end: aaa.loc[start:end],
    )

    recorder.capture(models_module.get_graham_intrinsic_value())
    recorder.capture(models_module.get_graham_intrinsic_value(0.05, 0.044))

    # By hand: the EPS of the period, its compound growth since the first period (from
    # three years of history, kept within 0 and 0.20) and the yield on the last business
    # day up to the period end.
    income = models_module._income_statement
    earnings_per_share = (
        income.loc[:, "Net Income", :]
        / income.loc[:, "Weighted Average Shares Diluted", :]
    )
    value = models_module.get_graham_intrinsic_value(rounding=10)
    market_dates = models_module._market_dates()
    checked = 0
    for ticker in value.index:
        history = earnings_per_share.loc[ticker].dropna()
        for position, period in enumerate(history.index[1:], start=1):
            if period not in value.columns:
                continue
            if position < 3:
                assert pd.isna(value.loc[ticker, period])
                continue
            growth = min(
                max((history[period] / history.iloc[0]) ** (1 / position) - 1, 0), 0.2
            )
            bond_yield = aaa.loc[: market_dates.loc[ticker, period]].iloc[-1]
            expected = history[period] * (8.5 + 200 * growth) * 4.4 / (bond_yield * 100)
            assert value.loc[ticker, period] == pytest.approx(expected, rel=1e-9)
            checked += 1
    assert checked >= 2


def test_graham_intrinsic_value_without_a_bond_yield_source(models_module, monkeypatch):
    monkeypatch.setattr(models_module, "_corporate_bond_yield_function", None)

    # Without a source the bond yield must be given.
    assert models_module.get_graham_intrinsic_value().empty
    assert not models_module.get_graham_intrinsic_value(bond_yield=0.05).empty


@pytest.mark.parametrize(
    ("quarterly", "trailing"), [(False, None), (False, 2), (True, None), (True, 4)]
)
def test_dupont_analyses_multiply_to_the_reported_return_on_equity(quarterly, trailing):
    from financetoolkit.models.models_controller import Models  # noqa: PLC0415
    from financetoolkit.ratios.ratios_controller import Ratios  # noqa: PLC0415

    statements = [
        pd.read_pickle(f"tests/datasets/{name}_dataset.pickle")
        for name in ("balance", "income", "cash")
    ]
    if quarterly:
        # Eight quarters out of the four years of the datasets, so that a trailing year of
        # four quarters, which averages the balances over five quarter ends, has values.
        statements = [
            pd.concat([statement, statement * 1.07], axis=1).set_axis(
                pd.period_range("2019Q1", periods=8, freq="Q"), axis=1
            )
            for statement in statements
        ]

    tickers = ["AAPL", "MSFT"]
    models = Models(tickers, {}, {}, *statements, quarterly=quarterly, rounding=None)
    ratios = Ratios(tickers, {}, *statements, quarterly=quarterly, rounding=None)

    return_on_equity = ratios.get_return_on_equity(trailing=trailing)
    asset_turnover = ratios.get_asset_turnover_ratio(trailing=trailing)
    return_on_assets = ratios.get_return_on_assets(trailing=trailing)
    dupont = models.get_dupont_analysis(trailing=trailing)
    extended = models.get_dupont_analysis(method="extended", trailing=trailing)

    for ticker in tickers:
        reported = return_on_equity.loc[ticker]
        assert reported.notna().sum() >= 2

        margin = dupont.loc[(ticker, "Net Profit Margin")]
        turnover = dupont.loc[(ticker, "Asset Turnover")]
        product = margin * turnover * dupont.loc[(ticker, "Equity Multiplier")]
        extended_product = (
            extended.loc[(ticker, "Interest Burden Ratio")]
            * extended.loc[(ticker, "Tax Burden Ratio")]
            * extended.loc[(ticker, "Operating Profit Margin")]
            * extended.loc[(ticker, "Asset Turnover")]
            * extended.loc[(ticker, "Equity Multiplier")]
        )

        for decomposed in (
            product,
            extended_product,
            dupont.loc[(ticker, "Return on Equity")],
            extended.loc[(ticker, "Return on Equity")],
        ):
            pd.testing.assert_series_equal(
                decomposed, reported, check_names=False, rtol=1e-12, atol=0
            )
        pd.testing.assert_series_equal(
            turnover, asset_turnover.loc[ticker], check_names=False, rtol=1e-12
        )
        pd.testing.assert_series_equal(
            margin * turnover,
            return_on_assets.loc[ticker],
            check_names=False,
            rtol=1e-12,
            atol=0,
        )


@pytest.mark.parametrize(
    ("quarterly", "trailing"), [(False, None), (False, 2), (True, None), (True, 4)]
)
def test_economic_value_added_uses_the_invested_capital_of_the_roic(
    quarterly, trailing
):
    from financetoolkit.models.models_controller import Models  # noqa: PLC0415
    from financetoolkit.ratios.ratios_controller import Ratios  # noqa: PLC0415

    statements = [
        pd.read_pickle(f"tests/datasets/{name}_dataset.pickle")
        for name in ("balance", "income", "cash")
    ]
    if quarterly:
        statements = [
            pd.concat([statement, statement * 1.07], axis=1).set_axis(
                pd.period_range("2019Q1", periods=8, freq="Q"), axis=1
            )
            for statement in statements
        ]

    tickers = ["AAPL", "MSFT"]
    models = Models(tickers, {}, {}, *statements, quarterly=quarterly, rounding=None)
    ratios = Ratios(tickers, {}, *statements, quarterly=quarterly, rounding=None)
    # The WACC needs market data and does not enter the NOPAT or the invested capital.
    wacc = pd.DataFrame(
        0.08,
        index=pd.MultiIndex.from_product(
            [tickers, ["Weighted Average Cost of Capital"]]
        ),
        columns=statements[0].columns,
    )
    models._weighted_average_cost_of_capital = lambda **_: wacc

    eva = models.get_economic_value_added(trailing=trailing)
    return_on_invested_capital = ratios.get_return_on_invested_capital(
        trailing=trailing
    )

    for ticker in tickers:
        reported = return_on_invested_capital.loc[ticker]
        assert reported.notna().sum() >= 2
        pd.testing.assert_series_equal(
            eva.loc[(ticker, "Net Operating Profit After Taxes")]
            / eva.loc[(ticker, "Invested Capital")],
            reported,
            check_names=False,
            rtol=1e-12,
            atol=0,
        )


def test_get_discretionary_accruals(recorder, models_module):
    # The offline statements cover too few years (and tickers) to estimate on, so only the
    # Total Accruals have values; the estimates are tested on synthetic statements below.
    recorder.capture(models_module.get_discretionary_accruals())


def test_discretionary_accruals_estimate_on_the_same_quarter_of_earlier_years():
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(7)
    quarters = 4 * 14
    assets = 1000 * np.cumprod(1 + rng.normal(0.01, 0.03, quarters))
    revenue = assets * rng.uniform(0.1, 0.3, quarters)
    fixed_assets = assets * rng.uniform(0.2, 0.6, quarters)
    # Accruals that follow the Jones model exactly on trailing twelve months and changes
    # over a year: a1 + a2 ΔRevenue + a3 PPE, spread evenly over the quarters.
    yearly_revenue = np.convolve(revenue, np.ones(4))[: len(revenue)]
    change = np.full(quarters, np.nan)
    change[4:] = yearly_revenue[4:] - yearly_revenue[:-4]
    yearly_accruals = 30 + 0.04 * change + -0.06 * fixed_assets
    quarterly_accruals = np.full(quarters, np.nan)
    quarterly_accruals[:4] = 1.0
    for quarter in range(4, quarters):
        quarterly_accruals[quarter] = (
            yearly_accruals[quarter] - quarterly_accruals[quarter - 3 : quarter].sum()
        )

    models = synthetic_models(
        {
            "Total Assets": {"A": list(assets)},
            "Revenue": {"A": list(revenue)},
            "Accounts Receivable": {"A": [50.0] * quarters},
            "Property, Plant and Equipment": {"A": list(fixed_assets)},
            "Net Income": {"A": list(quarterly_accruals)},
            "Operating Cash Flow": {"A": [0.0] * quarters},
        },
        quarterly=True,
    )
    result = models.get_discretionary_accruals().loc["A"]
    discretionary = result.loc["Discretionary Accruals"].dropna()
    # The first change of trailing twelve months is in the eighth quarter (position 7),
    # so the first estimate on ten earlier years of the same quarter is at position 47.
    assert len(discretionary) == quarters - (7 + 4 * 10)
    assert np.abs(discretionary.to_numpy()).max() < 1e-10
    # Accruals that follow the model without a constant give a zero constant when one is added.
    with_intercept = models.get_discretionary_accruals(intercept=True).loc["A"]
    np.testing.assert_allclose(
        with_intercept.loc["Discretionary Accruals"].dropna(), 0, atol=1e-10
    )
    assert with_intercept.loc["Discretionary Accruals"].notna().sum() == len(
        discretionary
    )

    cross_section = models.get_discretionary_accruals(method="cross_section")
    assert cross_section.loc["A"].loc["Discretionary Accruals"].isna().all()
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        models.get_discretionary_accruals(method="panel")


def test_quarterly_intrinsic_valuation_sums_the_quarters_of_a_year():
    import numpy as np  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    g = GROWTH ** np.arange(QUARTERS)
    settings = {
        "growth_rate": 0.0,
        "perpetual_growth_rate": 0.02,
        "weighted_average_cost_of_capital": 0.09,
        "cash_flow_type": "Free Cash Flow",
    }

    # Every quarter reported: the base is the trailing twelve months.
    models = synthetic_models(growing_company())
    valuation = models.get_intrinsic_valuation(**settings)
    assert valuation.loc["A"].iloc[:, -1]["Cash Flow Projection"] == pytest.approx(
        20 * g[-4:].sum()
    )

    # The latest quarter but one is not reported (no column at all): the base is the latest
    # four quarters in a row, not the latest four reported columns spanning five quarters.
    models = synthetic_models(growing_company())
    kept = models._cash_flow_statement.columns.delete(QUARTERS - 2)
    models._balance_sheet_statement = models._income_statement = (
        models._cash_flow_statement
    ) = models._cash_flow_statement.loc[:, kept]
    valuation = models.get_intrinsic_valuation(**settings)
    assert valuation.loc["A"].iloc[:, -1]["Cash Flow Projection"] == pytest.approx(
        20 * g[-6:-2].sum()
    )
    historical = models.get_intrinsic_valuation(historical=True, **settings)
    # The fourth quarter's year includes the missing third quarter: only the second
    # quarter's trailing year, the latest complete one, is valued last.
    assert str(historical.loc["A"].dropna().index[-1]) == "2023Q2"

    # A company reporting half-yearly (second and fourth quarters only) has no complete year
    # of quarters, so it is not valued instead of summing two years of cash flow.
    models = synthetic_models(growing_company())
    columns = models._cash_flow_statement.columns
    half_yearly = columns[columns.quarter.isin([2, 4])]
    models._balance_sheet_statement = models._income_statement = (
        models._cash_flow_statement
    ) = models._cash_flow_statement.loc[:, half_yearly]
    assert models.get_intrinsic_valuation(**settings).empty
    assert (
        models.get_intrinsic_valuation(historical=True, **settings).isna().all().all()
    )


def test_get_fundamental_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_fundamental_growth_rate())
    recorder.capture(models_module.get_fundamental_growth_rate(growth=True))


def test_get_rule_of_40(recorder, models_module):
    recorder.capture(models_module.get_rule_of_40())
    recorder.capture(models_module.get_rule_of_40(margin="ebitda"))
    with pytest.raises(InvalidArgumentError, match="margin must be one of"):
        models_module.get_rule_of_40(margin="operating")


def test_get_owner_earnings(recorder, models_module):
    recorder.capture(models_module.get_owner_earnings())
    recorder.capture(models_module.get_owner_earnings(method="greenwald"))


def test_get_return_on_incremental_invested_capital(recorder, models_module):
    recorder.capture(models_module.get_return_on_incremental_invested_capital())
    recorder.capture(models_module.get_return_on_incremental_invested_capital(years=2))


def test_get_unlevered_beta(recorder, models_module):
    recorder.capture(models_module.get_unlevered_beta())
    recorder.capture(models_module.get_unlevered_beta(rounding=2))


def test_get_capitalized_research_and_development(recorder, models_module):
    recorder.capture(
        models_module.get_capitalized_research_and_development(amortization_years=2)
    )
    recorder.capture(
        models_module.get_capitalized_research_and_development(amortization_years=1)
    )
    # The four years of the offline statements are too few for the default five-year life.
    default = models_module.get_capitalized_research_and_development()
    assert default.xs("Research Asset", level=1).isna().all().all()


def test_get_cash_flow_return_on_investment(recorder, models_module):
    recorder.capture(models_module.get_cash_flow_return_on_investment())
    recorder.capture(models_module.get_cash_flow_return_on_investment(rounding=2))


def test_get_beneish_five_variable_m_score(recorder, models_module):
    five = models_module.get_beneish_m_score(method="five", rounding=10)
    recorder.capture(five)
    eight = models_module.get_beneish_m_score(rounding=10)[five.columns]
    indices = eight.xs("Days Sales in Receivables Index", level=1)
    pd.testing.assert_frame_equal(
        five.xs("Days Sales in Receivables Index", level=1), indices
    )
    expected = (
        -6.065
        + 0.823 * indices
        + 0.906 * eight.xs("Gross Margin Index", level=1)
        + 0.593 * eight.xs("Asset Quality Index", level=1)
        + 0.717 * eight.xs("Sales Growth Index", level=1)
        + 0.107 * eight.xs("Depreciation Index", level=1)
    )
    pd.testing.assert_frame_equal(
        five.xs("Beneish M-Score", level=1), expected, atol=1e-8
    )
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        models_module.get_beneish_m_score(method="six")


def test_get_shumway_score(recorder, models_module, monkeypatch):
    # The S&P 500 offline, as in the Campbell-Hilscher-Szilagyi test.
    benchmark = models_module._historical_data["daily"]["Adj Close"]["Benchmark"]
    benchmark.index = benchmark.index.to_timestamp()

    def sp500(start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:  # noqa: ARG001
        index_data = pd.DataFrame(
            {"Price Index": benchmark, "Total Return Index": benchmark}
        )
        return index_data, pd.Series({benchmark.index[-1]: 3e13})

    monkeypatch.setattr(models_module, "_sp500_function", sp500)
    recorder.capture(models_module.get_shumway_score())
    recorder.capture(models_module.get_shumway_score(method="market"))

    # Without the S&P 500, the relative size and the probability are NaN.
    monkeypatch.setattr(models_module, "_sp500_function", None)
    result = models_module.get_shumway_score()
    assert result.xs("Bankruptcy Probability", level=1).isna().all().all()


def test_get_graham_criteria(recorder, models_module):
    recorder.capture(models_module.get_graham_criteria())
    recorder.capture(models_module.get_graham_criteria(minimum_revenue=3e11))


def long_history_models():
    """A company with 25 years of statements, for the measures that need long histories."""
    rng = np.random.default_rng(7)
    years = 25
    trend = 1.05 ** np.arange(years)
    net_income = 60 * trend + 10 * rng.standard_normal(years)
    items = {
        "Total Assets": 1000 * trend * (1 + 0.05 * rng.standard_normal(years)),
        "Operating Cash Flow": 80 * trend + 15 * rng.standard_normal(years),
        "Change in Working Capital": 10 * rng.standard_normal(years),
        "Net Income": net_income,
        "Net Income from Continuing Operations": net_income,
        "Weighted Average Shares": np.full(years, 10.0),
        "Revenue": 900 * 1.06 ** np.arange(years) + 30 * rng.standard_normal(years),
        "Property, Plant and Equipment": 300 * 1.04 ** np.arange(years),
    }
    return synthetic_models(
        {row: {"A": list(values)} for row, values in items.items()}, quarterly=False
    )


def test_get_accrual_quality(recorder, models_module):
    # The offline statements are too short for a regression of eight years.
    short = models_module.get_accrual_quality()
    assert short.xs("Accrual Quality", level=1).isna().all().all()
    models = long_history_models()
    recorder.capture(models.get_accrual_quality())
    recorder.capture(models.get_accrual_quality(method="mcnichols", window=10))


def test_get_earnings_attributes(recorder):
    models = long_history_models()
    recorder.capture(models.get_earnings_attributes())
    recorder.capture(models.get_earnings_attributes(window=5))


def _offline_toolkit(income=None, cash=None):
    """The offline Toolkit of the fixtures, with statements changed by the test."""
    from financetoolkit import Toolkit  # noqa: PLC0415

    def read(name):
        return pd.read_pickle(f"tests/datasets/{name}_dataset.pickle")

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=read("historical"),
        balance=read("balance"),
        income=read("income") if income is None else income,
        cash=read("cash") if cash is None else cash,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
        use_cached_data=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")
    return toolkit


def test_zero_shares_leave_the_value_per_share_out_for_that_ticker_only():
    income = pd.read_pickle("tests/datasets/income_dataset.pickle")
    for row in ("Weighted Average Shares", "Weighted Average Shares Diluted"):
        income.loc[("MSFT", row)] = 0.0
    models = _offline_toolkit(income=income).models

    earnings_power_value = models.get_earnings_power_value()
    assert earnings_power_value.loc["AAPL"].notna().all()
    assert np.isnan(earnings_power_value.loc["MSFT", "Earnings Power Value per Share"])

    intrinsic_value = models.get_intrinsic_valuation().xs("Intrinsic Value", level=1)
    assert np.isfinite(intrinsic_value.loc["AAPL"]).all()
    assert intrinsic_value.loc["MSFT"].isna().all()

    growth_opportunities = models.get_present_value_of_growth_opportunities()
    assert not np.isinf(growth_opportunities.to_numpy(dtype=float)).any()
    assert growth_opportunities["AAPL"].notna().any()

    simulated = models.get_monte_carlo_intrinsic_valuation(simulations=100)
    assert not np.isinf(simulated.to_numpy(dtype=float)).any()
    assert simulated.loc["AAPL"].notna().all()


def test_a_period_without_revenue_has_no_margin():
    income = pd.read_pickle("tests/datasets/income_dataset.pickle")
    income.loc[("MSFT", "Revenue"), pd.Period("2023", "Y")] = 0.0
    models = _offline_toolkit(income=income).models

    dupont = models.get_dupont_analysis()
    assert not np.isinf(dupont.to_numpy(dtype=float)).any()
    assert np.isnan(dupont.loc[("MSFT", "Net Profit Margin"), "2023"])

    margin = models.get_earnings_power_value()["Average Operating Margin"]
    assert np.isfinite(margin).all()


def test_get_intrinsic_valuation_names_the_missing_free_cash_flow_to_firm(caplog):
    cash = pd.read_pickle("tests/datasets/cash_dataset.pickle")
    models = _offline_toolkit(
        cash=cash.drop(index="Capital Expenditure", level=1)
    ).models

    assert models.get_intrinsic_valuation().empty
    assert "free cash flow to firm is not available" in caplog.text


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("get_intrinsic_valuation", {"periods": 0}),
        ("get_monte_carlo_intrinsic_valuation", {"periods": -1}),
        ("get_monte_carlo_intrinsic_valuation", {"growth_rate_volatility": -0.5}),
        (
            "get_monte_carlo_intrinsic_valuation",
            {"weighted_average_cost_of_capital_volatility": -0.01},
        ),
        ("get_reverse_discounted_cash_flow", {"periods": 0}),
        ("get_earnings_power_value", {"periods": 0}),
        ("get_accrual_quality", {"window": 4}),
        ("get_accrual_quality", {"method": "mcnichols", "window": 6}),
        ("get_earnings_attributes", {"window": 2}),
        ("get_discretionary_accruals", {"min_observations": 2}),
        ("get_discretionary_accruals", {"min_observations": 3, "intercept": True}),
        ("get_mohanram_g_score", {"variance_periods": 1}),
        ("get_capitalized_research_and_development", {"amortization_years": 0}),
        ("get_return_on_incremental_invested_capital", {"years": 0}),
        ("get_factor_analysis", {"metrics": "bogus"}),
    ],
)
def test_arguments_outside_their_range_are_refused(models_module, method, arguments):
    with pytest.raises(InvalidArgumentError):
        getattr(models_module, method)(**arguments)


def test_measures_without_enough_years_say_so(models_module, caplog):
    assert models_module.get_earnings_attributes().empty
    assert "The Earnings Attributes could not be calculated" in caplog.text

    models_module.get_accrual_quality()
    assert "The Accrual Quality needs 10 years of statements" in caplog.text


def test_get_capitalized_research_and_development_without_research_says_so(caplog):
    income = pd.read_pickle("tests/datasets/income_dataset.pickle")
    models = _offline_toolkit(
        income=income.drop(index="Research and Development Expenses", level=1)
    ).models

    models.get_capitalized_research_and_development()
    assert "No Research and Development Expenses are reported for AAPL, MSFT" in (
        caplog.text
    )


def test_a_trailing_window_below_one_is_refused(models_module, caplog):
    for trailing in (0, -1, 2.5):
        with pytest.raises(InvalidArgumentError, match="trailing"):
            models_module.get_kaplan_zingales_index(trailing=trailing)

    with caplog.at_level("WARNING", logger="financetoolkit"):
        models_module.get_kaplan_zingales_index(trailing=100)
    assert caplog.text.count("The trailing window of 100 periods is longer") == 1


def test_a_reverse_discounted_cash_flow_after_the_prices_says_why(caplog):
    from financetoolkit import Toolkit

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2023-06-01",
        end_date="2023-12-31",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = toolkit.models.get_reverse_discounted_cash_flow()

    assert result["Implied Growth Rate"].isna().all()
    assert caplog.text.count("AAPL, MSFT has no share price between") == 1


def test_reverse_discounted_cash_flow_names_the_years_of_the_historical_growth():
    items = growing_company()
    items["Free Cash Flow"]["B"] = list(items["Free Cash Flow"]["A"])
    # B's first cash flow is negative, so it has no historical growth and no period.
    items["Free Cash Flow"]["B"][0] = -1.0
    for values in items.values():
        values.setdefault("B", list(values["A"]))
    reverse = synthetic_models(items, quarterly=False).get_reverse_discounted_cash_flow(
        weighted_average_cost_of_capital=0.09, cash_flow_type="Free Cash Flow"
    )
    assert reverse.loc["A", "Historical Growth Period"] == "2010-2029"
    assert np.isnan(reverse.loc["B", "Historical Growth Rate"])
    assert pd.isna(reverse.loc["B", "Historical Growth Period"])
    # The numbers keep their float dtype next to the period labels.
    assert reverse["Historical Growth Rate"].dtype == float
    assert reverse["Share Price"].dtype == float


def test_intrinsic_valuation_present_values_add_up_to_the_enterprise_value():
    valuation = synthetic_models(growing_company()).get_intrinsic_valuation(
        growth_rate=0.08,
        perpetual_growth_rate=0.025,
        weighted_average_cost_of_capital=0.1,
        cash_flow_type="Free Cash Flow",
        rounding=None,
    )
    a = valuation.loc["A"].iloc[:, -1]
    assert a["Present Value of Cash Flows"] + a[
        "Present Value of Terminal Value"
    ] == pytest.approx(a["Enterprise Value"], rel=1e-12)
    assert a["Present Value of Terminal Value"] == pytest.approx(
        a["Terminal Value"] / 1.1**5, rel=1e-12
    )
    assert list(valuation.loc["A"].index[2:5]) == [
        "Present Value of Cash Flows",
        "Present Value of Terminal Value",
        "Enterprise Value",
    ]
